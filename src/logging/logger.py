import os

import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import wandb


class BaseLogger(ABC):
    @abstractmethod
    def log_value(self, tag, value, step):
        pass

    @abstractmethod
    def log_value_dict(self, tag, d, step):
        pass

    @abstractmethod
    def log_histogram(self, tag, values, step):
        pass

    @abstractmethod
    def store(self):
        pass


class Logger(BaseLogger):
    """
    Class to handle all logging functionality. This should help simplify the process of
    switching to other logging services
    """

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_value(self, tag, value, step):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        self.writer.add_scalar(tag, value, step)

    def log_value_dict(self, tag, d, step):
        for k, v in d.items():
            key_tag = f"{tag}/{k}"
            if isinstance(v, dict):
                self.log_value_dict(key_tag, v, step)
            else:
                self.log_value(key_tag, v, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_params(self, model, tag_prefix, epoch):
        # Add weights as arrays to tensorboard
        for name, param in model.named_parameters():
            if param is not None:
                self.log_histogram(
                    tag=f"{tag_prefix}/param_{name}", values=param.cpu(), step=epoch
                )

    def log_grads(self, model, tag_prefix, epoch):
        # Add gradients as arrays to tensorboard
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(
                    tag=f"{tag_prefix}/grad_{name}", values=param.grad.cpu(), step=epoch
                )

    def store(self):
        self.writer.flush()


class WandbLogger(BaseLogger):
    def __init__(self) -> None:
        super().__init__()

    def log_value(self, tag, value, step):

        wandb.log({tag: value, "epoch": step})

    def log_value_dict(self, tag, d, step):
        for k, v in d.items():
            key_tag = f"{tag}/{k}"
            if isinstance(v, dict):
                self.log_value_dict(key_tag, v, step)
            else:
                self.log_value(key_tag, v, step)

    def log_histogram(self, tag, values, step):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError
