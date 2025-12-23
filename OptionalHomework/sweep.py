import torch
import wandb

from inference import test_inference_time
from train import load_model

sweep_config = {
    "method": "bayes",
    "metric": {"name": "inference_time", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [1, 4, 8, 16, 32, 64, 128]},
        "num_workers": {"values": [0, 2, 4]},
        "pin_memory": {"values": [True, False]},
    },
}

def run_inference_sweep():
    wandb.init()
    cfg = wandb.config

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = load_model()

    sequential_time, inference_time = test_inference_time(
        model=model,
        batch_size=cfg.batch_size,
        device=device,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    wandb.log({
        "sequential_time": sequential_time,
        "inference_time": inference_time,
    })


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="inference-speed-test")
    wandb.agent(sweep_id, function=run_inference_sweep, count=20)
