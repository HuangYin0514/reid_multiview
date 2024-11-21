from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):

    base.set_train()
    loader = loaders.loader
    meter = MultiItemAverageMeter()
    for i, data in enumerate(tqdm(loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.module == "Lucky":
            total_loss = base.model(imgs, pids, meter)

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

    return meter.get_val(), meter.get_str()
