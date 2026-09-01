import warnings
import torch.nn as nn

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.ddpl_h import DDPL_H   # changed


def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """

    if cfg.model.lower() == "ddpl_h":     ####hll
        model = DDPL_H(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
