# Annealed Winner-Takes-All for Motion Forecasting

[**Website**](https://valeoai.github.io/publications/awta/) | [**Paper**](https://arxiv.org/abs/2409.11172)

### 💡 In this work, we propose a plug-and-play loss that replaces the widely used Winner-Takes-All loss for motion forecasting models.
<div align="center">
  <img src=awta.png width="800px" />
</div>

<br /><br />
**annealed Winner-Takes-All (aWTA), a better loss for training motion forecasting models:** <br />
<div align="center">
  <img src="https://github.com/valeoai/MF_aWTA/blob/main/reg_loss_with_temp.gif" width="800px" />
</div>

🔥Powered by [Hydra](https://hydra.cc/docs/intro/), [Pytorch-lightinig](https://lightning.ai/docs/pytorch/stable/),
and [WandB](https://wandb.ai/site), the framework is easy to configure, train and log.


## 🛠 Quick Start (from Unitraj)

0. Create a new conda environment

```bash
conda create -n unitraj python=3.9
conda activate unitraj
```

1. Install ScenarioNet: https://scenarionet.readthedocs.io/en/latest/install.html
```bash
pip --no-cache-dir install "metadrive-simulator>=0.4.1.1"
python -m metadrive.examples.profile_metadrive # test your installation
cd ~/workspace/scenarionet
sudo apt-get update
sudo apt install libspatialindex-dev
pip --no-cache-dir install -e .
pip install --no-cache-dir av2 --upgrade  
```



2. Install Unitraj:

```bash
git clone https://github.com/valeoai/MF_aWTA
pip install -r requirements.txt
pip install  --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python setup.py develop #only for the first install
export PYTHONPATH=$PYTHONPATH:~/workspace/UniTraj
export PYTHONPATH=$PYTHONPATH:~/workspace/UniTraj/unitraj/models/mtr/ops

```
known issues:
1. Make sure to have the compiled `knn_cuda.cpython-39-x86_64-linux-gnu.so` in `~/workspace/UniTraj/unitraj/models/mtr/ops/knn` Otherwise, it means that the command `python setup.py develop` didn’t work well when 
install UniTraj

2. if you have path issue when running 'train.py' or 'predict.py', you can try to insert the absolute path of unitraj and `~/workspace/UniTraj/unitraj/models/mtr/ops/knn` at the beginning of 'train.py' and 'predict.py'

```python
import os
sys.path.append("/Path/TO/UniTraj/unitraj/models/mtr/ops/")
sys.path.append("/Path/TO/UniTraj/")
```

You can verify the installation of UniTraj via running the training script:

```bash
cd unitraj
python train.py config-name=mtr_av2_awta
```

The model will be trained on several sample data.

## Code Structure

There are three main components in UniTraj: dataset, model and config.
The structure of the code is as follows:

```
unitraj
├── configs
│   ├── config.yaml
│   ├── method
│   │   ├── autobot.yaml
│   │   ├── MTR.yaml
│   │   ├── wayformer.yaml
├── datasets
│   ├── base_dataset.py
│   ├── autobot_dataset.py
│   ├── wayformer_dataset.py
│   ├── MTR_dataset.py
├── models
│   ├── autobot
│   ├── mtr
│   ├── wayformer
│   ├── base_model
├── utils
```

There is a base config, dataset and model class, and each model has its own config, dataset and model class that inherit
from the base class.

## Pipeline

### 1. Data Preparation

The code is modified from UniTraj. And UniTraj takes data from [ScenarioNet](https://github.com/metadriverse/scenarionet) as input. Process the data with
ScenarioNet in advance.

1. You need to download [Argoverse 2](https://www.argoverse.org/av2.html#download-link) and [Waymo Open Motion Datasest](https://waymo.com/open/download). 
2. Convert the data into ScenarioNet format:
- For Argoverse 2:

```bash
python -m scenarionet.convert_argoverse2 -d /path/to/your/database/train/ –raw_data_path /path/to/your/raw_data/train/
python -m scenarionet.convert_argoverse2 -d /path/to/your/database/val/ –raw_data_path /path/to/your/raw_data/val/
```
- For WOMD
```bash
python -m scenarionet.convert_waymo -d /path/to/your/database/training/ --raw_data_path /path/to/your/database/train/ --num_workers 64
python -m scenarionet.convert_waymo -d /path/to/your/database/validation/ --raw_data_path /path/to/your/database/val/ --num_workers 64
```


### 2. Configuration

UniTraj uses [Hydra](https://hydra.cc/docs/intro/) to manage configuration files.

Universal configuration file is located in `unitraj/config/config.yaml`.
Each model has its own configuration file in `unitraj/config/method/`, for
example, `unitraj/config/method/autobot.yaml`.

The configuration file is organized in a hierarchical structure, and the configuration of the model is inherited from
the universal configuration file.

#### Configuration Example

Please refer to config.yaml and method/mtr.yaml for more details.

### 3. Train
The configurations for each method and dataset are provided in `./configs`. The top 5 best models based on minFDE will be saved under `./lightning_logs` and tensorboard logs are also saved in the same folder (loss, metrics and some visualizations during training.)
For example, for running MTR with Argoverse 2, you can run (you may need to specify the paths of Argoverse 2 scenario data in  `./configs/mtr_av2_awta.yaml`):
```bash
cd unitraj
python train.py --config-name=mtr_av2_awta
```
By default, the model is trained with 8 GPUs, you can modify the number of GPUs in the corresponding config file like `mtr_av2_awta`, the batch size could be changed in `configs/method/MTR_wo_anchor.yaml`.

### 4. Evaluation

1. Download the checkpoints from the Release tagged model_weights and put them into  `./model_zoo/`. 
2. Run the evaluation, as an example, to evaluate MTR with av2, you can run:
```bash
cd unitraj
python predict.py --config-name=mtr_av2_awta_predict
```

### 5. Train your model with aWTA made easy
aWTA is a standalone loss that is compatible to all motion forecasting models that formally use the WTA loss. You only need to change the WTA loss into aWTA? Here is an example:
From WTA loss:
```bash
def wta_loss(prediction, gt, gt_valid_mask):
    '''
    prediction: predicted forecasts, of shape [batch, hypotheses, timesteps, 2]
    gt: ground-truth forecasting trajectory, of shape [batch, timesteps, 2]
    gt_valid_mask: ground-truth forecasting mask indicating the valid future steps, of shape [batch, timesteps]
    
    '''
    # compute prediction, gt distance, such as ADE
    distance = compute_ade(prediction, gt, gt_valid_mask)
    
    # select the prediction with the minimum distance to the ground truth
    nearest_hypothesis_idxs = distance.argmin(dim=-1) # [batch]
    nearest_hypothesis_bs_idxs = torch.arange(nearest_hypothesis_idxs.shape[0]).type_as(nearest_hypothesis_idxs) # [batch]
    
    # extract the L2 distance bewteen the selected hypothesis and gt
    loss_reg = distance[nearest_hypothesis_bs_idxs, nearest_hypothesis_idxs] # [batch]
    return loss_reg.mean() # mean over the batch
```
To aWTA loss:
```bash
def awta_loss(prediction, gt, gt_valid_mask, cur_temperature):
    '''
    prediction: predicted forecasts, of shape [batch, hypotheses, timesteps, 2]
    gt: ground-truth forecasting trajectory, of shape [batch, timesteps, 2]
    gt_valid_mask: ground-truth forecasting mask indicating the valid future steps, of shape [batch, timesteps]
    cur_temperature: current temperature for aWTA
    '''
    # compute prediction, gt distance, such as ADE
    distance = compute_ade(prediction, gt, gt_valid_mask)

    # calculate the weights q(t): softmin of the distance, controlled by the current temperature
    awta_weights = torch.softmax(-1.0*distance/cur_temperature, dim=-1).detach() # [batch, hypotheses]
    
    # weight the distance by awta weights
    loss_reg = distance * awta_weights # [batch, hypotheses]
    return loss_reg.sum(-1).mean() # sum over weighted hypotheses, and average over the batch

def temperature_scheduler(init_temperature, cur_epoch, exp_base):
    '''
    init_temperature: initial temperature
    cur_epoch: curent number of epochs
    exp_base: exponential scheduler base    
    '''
    return init_temperature*exp_base**cur_epoch
```


---

### For citation:

```
@article{xu2025awta,
  title={Annealed Winner-Takes-All for Motion Forecasting},
  author    = {Yihong Xu and
               Victor Letzelter and
               Mickaël Chen and
               \'{E}loi Zablocki and
               Matthieu Cord},
  journal = {under review},
  year = {2025}
}
```

### Acknoledgement

The code is modified from [UniTraj](https://github.com/vita-epfl/UniTraj) and [MTR](https://github.com/sshaoshuai/MTR).


