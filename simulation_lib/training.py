import copy
import os
import uuid
from collections.abc import Callable

from cyy_naive_lib.log import add_file_handler, log_debug, log_info
from cyy_naive_lib.time_counter import TimeCounter

from .algorithm_factory import get_worker_config
from .config import DistributedTrainingConfig
from .context import FederatedLearningContext
from .worker import Worker

# we use these environment variables to save memory in large-scale training
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["USE_THREAD_DATALOADER"] = "1"


def start_server(context: FederatedLearningContext, server_config: dict) -> dict:
    server = server_config["constructor"](
        extra_kwargs={
            "context": context,
        }
    )
    log_debug("context id %d", id(context))

    server.start()
    log_info("stop server")

    res: dict = {}
    if hasattr(server.algorithm, "shapley_values"):
        res["sv"] = server.algorithm.shapley_values
    res |= {"performance": server.performance_stat}
    return res


def run_worker(
    constructor: Callable,
    context: FederatedLearningContext,
) -> None:
    worker: Worker = constructor(context=context)
    worker.start()


def start_workers(
    context: FederatedLearningContext,
    worker_configs: list[dict],
) -> None:
    assert isinstance(context, FederatedLearningContext)
    assert worker_configs
    worker_funs: list[Callable] = []
    worker_kwargs: list[dict] = []

    for worker_config in worker_configs:
        worker_funs.append(run_worker)
        worker_kwargs.append({"worker_config": worker_config})
    log_debug(
        "run %s workers in the same process",
        len(worker_configs),
    )
    context.submit(worker_funs, kwargs_list=worker_configs)


tasks: dict = {}
task_results: dict = {}


def train(
    config: DistributedTrainingConfig,
    practitioners: None | set = None,
) -> int | None:
    # we need to deepcopy config for concurrent training
    config = copy.deepcopy(config)
    practitioners = copy.deepcopy(practitioners)
    config.reset_session()
    config.apply_global_config()
    timer = TimeCounter()
    task_id = None
    if practitioners is None:
        add_file_handler(config.log_file)
    else:
        task_id = uuid.uuid4().int + os.getpid()
    worker_config = get_worker_config(
        config, task_id=task_id, practitioners=practitioners
    )
    context = worker_config.pop("context")
    assert isinstance(context, FederatedLearningContext)
    server_config = worker_config.get("server", None)
    assert server_config is not None
    context.submit(
        [start_server],
        server_config=server_config,
    )
    for worker_configs in worker_config["worker"]:
        start_workers(context=context, worker_configs=worker_configs)
    if practitioners is not None:
        tasks[task_id] = {
            "process_pool": context.executor_pool,
            "practitioner_ids": {practitioner.id for practitioner in practitioners},
            "config": config,
        }
        task_results[task_id] = {}
        return task_id
    context.executor_pool.wait_results(timeout=None)
    context.executor_pool.shutdown(wait=True)
    log_info("training took %s seconds", timer.elapsed_milliseconds() / 1000)
    return None


def get_training_result(task_id: int, timeout: None | float = None) -> None | dict:
    task = tasks[task_id]
    process_pool = task["process_pool"]
    results, not_done = process_pool.wait_results(timeout=timeout)
    for result in results.values():
        if result is not None:
            task_results[task_id] |= result
    if not_done:
        return None
    process_pool.shutdown()
    tasks.pop(task_id)
    log_info("finish task %s", task_id)
    stats: dict = {}
    practitioner_ids = task["practitioner_ids"]
    config = task["config"]
    assert practitioner_ids is not None
    for k, v in task_results[task_id].items():
        if k != "sv":
            stats[k] = v
            continue
        sv_dict: dict = {}
        for round_number, tmp_sv_dict in v.items():
            sv_dict[round_number] = {}
            for practitioner_id, worker_id in zip(
                sorted(practitioner_ids), range(config.worker_number), strict=False
            ):
                sv_dict[round_number][practitioner_id] = tmp_sv_dict[worker_id]
        stats[k] = sv_dict
    task_results.pop(task_id)
    return stats
