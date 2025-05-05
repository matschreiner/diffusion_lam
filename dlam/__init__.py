import lovely_tensors as lt
import torch

from dlam import data, logger, mlops, model, samplers, scheduler, trainer

lt.monkey_patch()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
