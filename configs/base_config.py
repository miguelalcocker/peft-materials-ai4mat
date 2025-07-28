from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    """Configuration for model selection and PEFT setup"""
    model_name: str = "m3rg-iitd/matscibert"
    tokenizer_name: Optional[str] = None
    cache_dir: str = "./models/cache"
    
    # PEFT configuration
    peft_type: str = "LORA"  # LORA, QLORA, XLORA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Will be auto-detected
    
    # Training configuration
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # Hardware configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging"""
    project_name: str = "AI4Mat-PEFT-Materials-2025"
    experiment_name: str = "baseline"
    output_dir: str = "./results"
    logging_steps: int = 10
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

@dataclass
class DataConfig:
    """Configuration for dataset handling"""
    data_dir: str = "./data"
    max_samples: Optional[int] = None  # For quick testing
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data processing
    preprocessing_num_workers: int = 4
    dataloader_num_workers: int = 4
