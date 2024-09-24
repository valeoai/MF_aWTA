import sys
import os
print("You should modify the path in unitraj/train.py. Hack implementation, by sys.path.insert.")
sys.path.append("./models/mtr/ops/")
sys.path.append("./")

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os
import wandb
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)
    #wandb.init()
    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)
    print("len(val_set) ",len(val_set))
    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size, 1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size, 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val--minFDE6',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=5,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    strategy = "auto"
    if not cfg.debug and len(cfg.devices) > 1:
        strategy = "ddp"
   
    #if cfg.ckpt_path is not None:
    #    cfg.method.max_epochs = 500
    #    print("cfg.method.max_epochs ", cfg.method.max_epochs)
    try:
        cfg.temperature_decay = cfg.temperature_decay
        cfg.temperature = cfg.temperature
    except:
        cfg.temperature_decay = 100
        cfg.temperature = 0.9
    logger = TensorBoardLogger("./lightning_logs", version=str(cfg.temperature_decay) + '_'+str(cfg.temperature), name=cfg.exp_name)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=logger, #WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        devices=cfg.devices if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy=strategy,
        callbacks=call_backs
    )
    
    if cfg.ckpt_path is not None:
        trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)
    # automatically resume training
    print("path", os.path.join('./lightning_logs', cfg.exp_name, str(cfg.temperature_decay) + '_'+str(cfg.temperature), 'checkpoints'))
    if cfg.ckpt_path is None and not cfg.debug:
        cfg.ckpt_path = find_latest_checkpoint(os.path.join('./lightning_logs', cfg.exp_name, str(cfg.temperature_decay) + '_'+str(cfg.temperature), 'checkpoints'))
    print("cfg.ckpt_path",cfg.ckpt_path)
    # load epoch count, useful for linear scheduler
    if cfg.ckpt_path is not None:
        #print("cfg.ckpt_path",cfg.ckpt_path)
        model.epoch_count = int(torch.load(cfg.ckpt_path)['epoch'])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
