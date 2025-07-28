import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import psutil
import os

def test_model_memory(model_name, description):
    """Test memory usage for specific model"""
    print(f"\n=== Testing {description} ===")
    print(f"Model: {model_name}")
    
    # Get baseline GPU memory
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.memory_allocated() / 1e9
    
    try:
        # Load model
        model = AutoModel.from_pretrained(model_name)
        model = model.cuda()
        
        # Check memory usage
        model_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Model memory: {model_memory - baseline_memory:.2f} GB")
        
        # Test with LoRA
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"] if "bert" in model_name.lower() else ["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        
        peft_model = get_peft_model(model, lora_config)
        peft_memory = torch.cuda.memory_allocated() / 1e9
        print(f"PEFT memory: {peft_memory - baseline_memory:.2f} GB")
        print(f"Memory overhead: {peft_memory - model_memory:.2f} GB")
        
        trainable_params = peft_model.num_parameters(only_trainable=True)
        total_params = peft_model.num_parameters()
        print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return {
            'model_memory': model_memory - baseline_memory,
            'peft_memory': peft_memory - baseline_memory,
            'trainable_params': trainable_params,
            'total_params': total_params
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if 'model' in locals():
            del model
        if 'peft_model' in locals():
            del peft_model
        torch.cuda.empty_cache()

# Test different models relevant to materials science
models_to_test = [
    ("distilbert-base-uncased", "DistilBERT (baseline)"),
    ("m3rg-iitd/matscibert", "MatSciBERT (materials science)"),
    ("DeepChem/ChemBERTa-77M-MLM", "ChemBERTa-77M (chemistry)"),
]

results = {}
for model_name, description in models_to_test:
    results[model_name] = test_model_memory(model_name, description)

print("\n=== SUMMARY ===")
for model_name, result in results.items():
    if result:
        print(f"{model_name}: {result['peft_memory']:.2f} GB (PEFT)")
