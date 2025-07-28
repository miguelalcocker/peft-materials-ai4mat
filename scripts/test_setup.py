import torch
import transformers
import peft
import wandb
import os
from datetime import datetime

def test_gpu_setup():
    """Test GPU availability and memory"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x.T)
        print("GPU tensor operations: OK")
        del x, y
        torch.cuda.empty_cache()
    
def test_wandb():
    """Test wandb functionality"""
    wandb.init(project="AI4Mat-PEFT-Materials-2025", 
               name=f"setup-test-{datetime.now().strftime('%Y%m%d-%H%M')}")
    wandb.log({"test_metric": 42})
    wandb.finish()
    print("Wandb logging: OK")

def test_transformers_peft():
    """Test basic PEFT functionality"""
    from transformers import AutoModel, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    
    # Test with small model
    model_name = "distilbert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    
    # Test LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
    )
    
    peft_model = get_peft_model(model, lora_config)
    print(f"PEFT model created. Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
    print("PEFT basic functionality: OK")

if __name__ == "__main__":
    print("=== Testing GPU Setup ===")
    test_gpu_setup()
    
    print("\n=== Testing Wandb ===")
    test_wandb()
    
    print("\n=== Testing PEFT ===")
    test_transformers_peft()
    
    print("\n✅ All systems functional!")

