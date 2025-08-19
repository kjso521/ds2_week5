"""
#  Copyright Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#
"""

import os
import time
from collections.abc import Callable
from dataclasses import asdict
from enum import Enum
from pathlib import Path

import torch
from scipy.io import savemat
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from common.logger import logger
from common.metric import calculate_psnr, calculate_ssim
from common.utils import seconds_to_dhms, validate_tensor_dimensions, validate_tensors
from components.metriccontroller import MetricController
from datawrapper.datawrapper import DataKey
from model.dncnn import DnCNN
from model.unet import Unet
from params import DnCNNConfig, UnetConfig, config, dncnnconfig, unetconfig

NETWORK = DnCNN | Unet | torch.nn.DataParallel[DnCNN] | torch.nn.DataParallel[Unet]
OPTIM = Adam | AdamW


class ModelType(str, Enum):
    DnCNN = "dncnn"
    Unet = "unet"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid ModelType value: {value}. Must be one of {list(cls)} : {err}") from err


def get_model(
    config: "GeneralConfig",
) -> NETWORK:
    device = config.device
    model_type = config.model_type

    if device is None:
        raise TypeError("device is not to be None")

    if ModelType.from_string(model_type) == ModelType.DnCNN:
        return DnCNN(
            channels=dncnnconfig.channels,
            num_of_layers=dncnnconfig.num_of_layers,
            kernel_size=dncnnconfig.kernel_size,
            padding=dncnnconfig.padding,
            features=dncnnconfig.features,
        )
    elif ModelType.from_string(model_type) == ModelType.Unet:
        return Unet(
            in_chans=unetconfig.in_chans,
            out_chans=unetconfig.out_chans,
            chans=unetconfig.chans,
            num_pool_layers=unetconfig.num_pool_layers,
        )
    else:
        raise KeyError("model type not matched")


def get_optimizer(
    config: "GeneralConfig",
    params: "torch.nn.Module.parameters",
) -> OPTIM | None:
    optimizer_type = config.optimizer
    
    if params is None:
        return None
    if optimizer_type == "adam":
        return Adam(params, lr=config.lr, betas=(0.9, 0.99))
    elif optimizer_type == "adamw":
        return AdamW(params, lr=config.lr, betas=(0.9, 0.99), weight_decay=0.0)
    else:
        raise KeyError("optimizer not matched")


def get_loss_model(
    config: "GeneralConfig",
) -> "torch.nn.Module":
    loss_model_type = config.loss_model
    if loss_model_type == "l1":
        return torch.nn.L1Loss()
    elif loss_model_type == "l2":
        return torch.nn.MSELoss()
    else:
        raise KeyError("loss func not matched")


def get_learning_rate(
    epoch: int,
    lr: float,
    lr_decay: float,
    lr_tol: int,
) -> float:
    factor = epoch - lr_tol if lr_tol < epoch else 0
    return lr * (lr_decay**factor)


def set_optimizer_lr(
    optimizer: OPTIM | None,
    learning_rate: float,
) -> OPTIM | None:
    if optimizer is None:
        return None
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return optimizer


def log_summary(
    init_time: float,
    state: MetricController,
    log_std: bool = False,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            logger.info(summary)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)


def save_checkpoint(
    network: NETWORK,
    run_dir: Path,
    epoch: str | int | None = None,
) -> None:
    if epoch is None:
        epoch = "best"
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "model_config": asdict(dncnnconfig),
        },
        run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
    )


def zero_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.zero_grad()


def step_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.step()


def save_result_to_mat(
    test_dir: Path,
    batch_cnt: int,
    tesner_dict: dict[str, Tensor | None],
    img_cnt: int,
) -> None:
    os.makedirs(test_dir, exist_ok=True)
    save_dict = {}

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        for key, value in tesner_dict.items():
            if value is not None:
                save_dict[key] = value.cpu().detach().numpy()[i, ...]

        idx = img_cnt + i + 1
        savemat(f"{test_dir}/{idx}_res.mat", save_dict)


def train_epoch_dncnn(
    _data: dict[DataKey, Tensor | str],
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
) -> int:
    loss_func = get_loss_model(config)

    label: Tensor = _data[DataKey.Label].to(config.device)

    img_cnt_minibatch = label.shape[0]

    output = network.forward(_data[DataKey.Noisy].to(config.device))

    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)

    torch.mean(loss).backward()
    train_state.add("loss", loss)

    return img_cnt_minibatch


def train_epoch(
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optim_list: list[OPTIM | None],
    epoch: int,
) -> None:
    train_state = MetricController()
    train_state.reset()
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    for _data in train_loader:
        zero_optimizers(optim_list=optim_list)
        if ModelType.from_string(config.model_type) == ModelType.DnCNN:
            img_cnt_minibatch = train_epoch_dncnn(
                _data=_data,
                network=network,
                epoch=epoch,
                train_state=train_state,
            )
        else:
            raise KeyError("model type not matched")

        step_optimizers(optim_list=optim_list)
        img_cnt += img_cnt_minibatch
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1

    log_summary(init_time=config.init_time, state=train_state)


def test_part_dncnn(
    _data: dict[DataKey, Tensor | str],
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
) -> float:
    noisy = _data[DataKey.Noisy].to(config.device)
    label = _data[DataKey.Label].to(config.device)

    batch_cnt = noisy.shape[0]

    validate_tensors([noisy, label])
    validate_tensor_dimensions([noisy, label], 4)

    output = model(noisy)

    validate_tensors([output])
    validate_tensor_dimensions([output], 4)

    for idx in range(output.shape[0]):
        test_state.add("psnr", calculate_psnr(output[idx : idx + 1, ...], label[idx : idx + 1, ...]))
        test_state.add("ssim", calculate_ssim(output[idx : idx + 1, ...], label[idx : idx + 1, ...]))

    if save_val:
        save_result_to_mat(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "noisy": noisy,
                "output": output,
                "label": label,
            },
            img_cnt=img_cnt,
        )

    return batch_cnt


def test_part(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
) -> float:
    test_state = MetricController()
    test_state.reset()
    network.eval()
    model = network.module if isinstance(network, torch.nn.DataParallel) else network

    img_cnt: int = 0
    for _data in data_loader:
        if ModelType.from_string(config.model_type) == ModelType.DnCNN:
            batch_cnt = test_part_dncnn(
                _data=_data,
                test_dir=run_dir / f"test/ep_{epoch}",
                model=model,
                save_val=save_val and img_cnt <= config.save_max_idx,
                test_state=test_state,
                img_cnt=img_cnt,
            )
        else:
            raise KeyError("model type not matched")

        img_cnt += batch_cnt

    log_summary(init_time=config.init_time, state=test_state, log_std=True)

    primary_metric = test_state.mean("psnr")
    return primary_metric
