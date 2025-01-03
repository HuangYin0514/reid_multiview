import argparse
import ast
import os
import random
import warnings

import numpy as np
import torch
import wandb
from core import Base, test, train, visualization
from data_loader.loader import Loader
from tools import Logger, make_dirs, os_walk, time_now

warnings.filterwarnings("ignore")


best_mAP = 0
best_rank1 = 0
best_epoch = 0


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    print("#" * 10)
    print("Main running...")

    global best_mAP
    global best_rank1
    global best_epoch

    print("Loading data...")
    loaders = Loader(config)
    print("Loading model...")
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, "logs/"), "logger.log"))
    logger("\n" * 3)
    logger(config)

    if config.mode == "train":
        ########################################################
        # 训练
        ########################################################

        # wandb.watch(model.model, log="all")

        # 恢复训练
        if config.resume_train_epoch >= 0:
            model.resume_model(config.resume_train_epoch)
            start_train_epoch = config.resume_train_epoch
        else:
            start_train_epoch = 0

        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(model.save_model_path)
            print(files)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace(".pth", "").split("_")[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                model.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger("Time: {}, automatically resume training from the latest step (model {})".format(time_now(), indexes[-1]))

        # 训练
        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            model.model_lr_scheduler.step(current_epoch)

            if current_epoch < config.total_train_epoch:
                dict_result, result = train(model, loaders, config)
                logger("Time: {}; Epoch: {}; {}".format(time_now(), current_epoch, result))
                wandb.log(
                    {
                        "Lr": model.model_optimizer.param_groups[0]["lr"],
                        **dict_result,
                    }
                )

            if current_epoch + 1 >= 1 and (current_epoch + 1) % config.eval_epoch == 0:
                mAP, CMC = test(config, model, loaders)
                is_best_rank = CMC[0] >= best_rank1
                if is_best_rank:
                    best_epoch = current_epoch
                    best_rank1 = CMC[0]
                    best_mAP = mAP
                    wandb.log(
                        {
                            "best_epoch": best_epoch,
                            "best_rank1": best_rank1,
                            "best_mAP": best_mAP,
                        }
                    )
                model.save_model(current_epoch, is_best_rank)
                logger("Time: {}; Test on Dataset: {}, \nmAP: {} \nRank: {}".format(time_now(), config.test_dataset, mAP, CMC))
                wandb.log(
                    {
                        "test_epoch": current_epoch,
                        "mAP": mAP,
                        "Rank1": CMC[0],
                        "Rank5": CMC[4],
                        "Rank10": CMC[9],
                        "Rank20": CMC[19],
                    }
                )

        if best_rank1:
            logger("=" * 50)
            logger("Best model is: epoch: {}, rank1 {}".format(best_epoch, best_rank1))
            logger("=" * 50)

    # 测试
    elif config.mode == "test":
        model.resume_model(config.resume_test_model)
        mAP, CMC = test(config, model, loaders)
        logger("Time: {}; Test on Dataset: {}, \nmAP: {} \n Rank: {}".format(time_now(), config.test_dataset, mAP, CMC))

    # 可视化
    elif config.mode == "visualization":
        loaders._visualization_load()
        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(model.save_model_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace(".pth", "").split("_")[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                model.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger("Time: {}, automatically resume training from the latest step (model {})".format(time_now(), indexes[-1]))
        visualization(config, model, loaders)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument("--task_name", type=str, default="kaggle version")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--cuda", type=str, default="cuda")
    parser.add_argument("--tags", type=str, nargs="+", help="master, dev, 对比实验")
    # Dataset
    parser.add_argument("--occluded_duke_path", type=str, default="/home/hy/project/data/Occluded_Duke")
    parser.add_argument("--occluded_reid_path", type=str, default="/opt/data/private/data/Occluded_REID")
    parser.add_argument("--partial_duke_path", type=str, default="/opt/data/private/data/P_Duke_OURS/new")
    parser.add_argument("--partial_reid_path", type=str, default="/opt/data/private/data/Partial-REID_Dataset")
    parser.add_argument("--partial_ilids_path", type=str, default="/opt/data/private/data/Partial_iLIDS")
    parser.add_argument("--market_path", type=str, default="/opt/data/private/data//Market-1501-v15.09.15")
    parser.add_argument("--duke_path", type=str, default="/opt/data/private/data/DukeMTMC-reID")
    parser.add_argument("--msmt_path", type=str, default="/opt/data/private/data/MSMT17")
    parser.add_argument("--train_dataset", type=str, default="occluded_duke", help="occluded_duke, occluded_reid, " "market, duke")
    parser.add_argument("--test_dataset", type=str, default="occluded_duke", help="occluded_duke, occluded_reid, " "market, duke")
    parser.add_argument("--pid_num", type=int, default=702)
    # Data loader
    parser.add_argument("--image_size", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--use_rea", type=ast.literal_eval, default=True, help="use random erasing augmentation")
    parser.add_argument("--use_colorjitor", type=ast.literal_eval, default=False, help="use random erasing augmentation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--num_instances", type=int, default=8)
    # Train
    parser.add_argument("--mode", type=str, default="train", help="train, test, visualization")
    parser.add_argument("--module", type=str, default="CIP", help="B, CIP_w_Q_L, CIP_w_L, CIP_w_Q, CIP")
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--milestones", nargs="+", type=int, default=[40, 70], help="milestones for the learning rate decay")
    parser.add_argument("--total_train_epoch", type=int, default=120)
    parser.add_argument("--eval_epoch", type=int, default=5)
    parser.add_argument("--resume_test_model", type=int, default=119, help="-1 for no resuming")
    parser.add_argument("--test_mode", type=str, default="inter-camera", help="inter-camera, intra-camera, all")
    # Save
    parser.add_argument("--output_path", type=str, default="occluded_duke/base/", help="path to save related informations")
    parser.add_argument("--max_save_model_num", type=int, default=1, help="0 for max num is infinit")
    parser.add_argument("--resume_train_epoch", type=int, default=-1, help="-1 for no resuming")
    parser.add_argument("--auto_resume_training_from_lastest_step", type=ast.literal_eval, default=True)
    config = parser.parse_args()
    seed_torch(config.seed)

    # 初始化wandb
    wandb.init(
        entity="yinhuang-team-projects",
        project="multi-view",
        name=config.task_name,
        notes=config.notes,
        tags=config.tags,
        config=config,
    )
    main(config)
    wandb.finish()