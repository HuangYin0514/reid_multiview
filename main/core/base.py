import os
from bisect import bisect_right

import torch
import torch.nn as nn
import torch.optim as optim
from network import CosineLRScheduler, Model
from tools import os_walk


class Base:
    def __init__(self, config):
        self.config = config

        self.pid_num = config.pid_num
        self.module = config.module

        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, "models/")
        self.save_logs_path = os.path.join(self.output_path, "logs/")

        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self._init_device()
        self._init_model()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device(self.config.cuda)

    def _init_model(self):
        self.model = Model(self.config)
        self.model = nn.DataParallel(self.model).to(self.device)

    def _init_optimizer(self):

        ########################################
        # Optimizer
        ########################################
        BASE_LR = 0.008
        WEIGHT_DECAY = 1e-4
        BIAS_LR_FACTOR = 2
        WEIGHT_DECAY_BIAS = 1e-4
        LARGE_FC_LR = False
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            lr = BASE_LR
            weight_decay = WEIGHT_DECAY
            if "bias" in key:
                lr = BASE_LR * BIAS_LR_FACTOR
                weight_decay = WEIGHT_DECAY_BIAS
            if LARGE_FC_LR:
                if "classifier" in key or "arcface" in key:
                    lr = BASE_LR * 2
                    print("Using two times learning rate for fc ")
            params += [
                {
                    "params": [value],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            ]
        MOMENTUM = 0.9
        self.model_optimizer = getattr(torch.optim, "SGD")(params, momentum=MOMENTUM)

        ########################################
        # Scheduler
        ########################################
        # self.model_lr_scheduler = WarmupMultiStepLR(self.model_optimizer, self.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        num_epochs = 120
        lr_min = 0.002 * BASE_LR
        warmup_lr_init = 0.01 * BASE_LR
        warmup_t = 5
        noise_range = None
        self.model_lr_scheduler = CosineLRScheduler(
            self.model_optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul=1.0,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
        )

    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, "model_{}.pth".format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if ".pth" not in file:
                    files.remove(file)
            if len(files) > 1 * self.max_save_model_num:
                file_iters = sorted([int(file.replace(".pth", "").split("_")[1]) for file in files], reverse=False)
                model_file_path = os.path.join(root, "model_{}.pth".format(file_iters[0]))
                os.remove(model_file_path)

    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if ".pth" not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace(".pth", "").split("_")[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        model_path = os.path.join(self.save_model_path, "model_{}.pth".format(resume_epoch))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print("Successfully resume model from {}".format(model_path))

    def set_train(self):
        self.model = self.model.train()
        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.training = False


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs]
