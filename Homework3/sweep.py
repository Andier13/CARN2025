from pathlib import Path
import yaml
import wandb
from train import main

if __name__ == '__main__':
    cfg_path = Path("configs/current_config.yaml")
    sweep_config: dict = yaml.safe_load(open(cfg_path))
    sweep_id = wandb.sweep(sweep=sweep_config, project="cifar100-project")
    wandb.agent(sweep_id, function=main, count=8)
