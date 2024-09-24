import sys
import os
print("You should modify the path in predict.py. Hack implementation, by sys.path.insert.")
sys.path.append("./models/mtr/ops/")
sys.path.append("./")
import torch

import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

from torch.utils.data import DataLoader

import hydra

from models import build_model
from datasets import build_dataset
from omegaconf import OmegaConf
from utils.utils import set_seed

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import BasePredictionWriter


@hydra.main(version_base=None, config_path="configs", config_name="config")
def prediction(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True
    my_path = cfg.ckpt_path 
    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_set.data_chunk_size, 1)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    data_path = cfg["val_data_path"]
    assert len(data_path) == 1 # below won't work if there are multiple files
    data_path = data_path[0]
    dataset_name = data_path.split('/')[-1]
    output_path = os.path.join(data_path, f'output_{cfg.method.model_name}')
  
    model.ckpt_name='/'.join(cfg.ckpt_path.split('/')[-2:]).replace('.ckpt','.pkl')
    print("model.ckpt_name ", model.ckpt_name)
    trainer = pl.Trainer(
        inference_mode=True,
        logger=None, # if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
        devices=[0], #cfg.devices,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy='ddp'
    )
    model.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'], strict=False)
    predictions = trainer.predict(model=model, dataloaders=val_loader)

if __name__ == '__main__':
    prediction()

