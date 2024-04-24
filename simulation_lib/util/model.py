from typing import Iterable

import torch.nn
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox.trainer import Trainer
from cyy_torch_toolbox.typing import TensorDict


def reset_optimizer_parameters(
    trainer: Trainer, params: Iterable[torch.nn.Parameter] | None = None
) -> None:
    optimizer = trainer.get_optimizer()
    assert len(optimizer.param_groups) == 1
    old_param_group = optimizer.param_groups[0]
    optimizer.param_groups.clear()
    optimizer.state.clear()
    if params is None:
        optimizer.add_param_group({"params": trainer.model.parameters()})
    else:
        optimizer.add_param_group({"params": params})
    for k, v in old_param_group.items():
        if k not in "params":
            optimizer.param_groups[0][k] = v
            log_debug("reuse parameter property %s", k)


def load_parameters(
    trainer: Trainer, parameter_dict: TensorDict, reuse_learning_rate: bool
) -> None:
    if reuse_learning_rate:
        trainer.model_util.load_parameter_dict(parameter_dict)
        reset_optimizer_parameters(trainer=trainer)
    else:
        trainer.load_parameter_dict(parameter_dict)
    trainer.model_util.disable_running_stats()
