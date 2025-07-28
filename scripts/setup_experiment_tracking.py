import wandb
import torch
import random
import numpy as np
from configs.base_config import ExperimentConfig

def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def init_wandb_tracking(config: ExperimentConfig):
    """Initialize wandb tracking"""
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        config=config.__dict__
    )
    return wandb

if __name__ == "__main__":
    config = ExperimentConfig()
    set_seed(config.seed)
    
    # Test wandb initialization
    wb_run = init_wandb_tracking(config)
    wandb.log({"setup_test": 1})
    wandb.finish()
    print("Experiment tracking setup complete!")
