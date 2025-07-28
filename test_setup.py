# ============================================
# VERIFICATION TEST - Run this after setup
# Save as: ~/peft-scientific/test_setup.py
# ============================================

import torch
import transformers
import peft
import datasets
import accelerate
import bitsandbytes
import sys
import psutil

def test_setup():
    print("🧪 PEFT Scientific Setup Verification")
    print("=" * 50)
    
    # 1. Python Environment
    print(f"🐍 Python Version: {sys.version}")
    print(f"💾 Available RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # 2. GPU Check
    print(f"\n🖥️  GPU STATUS:")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA Available: True")
        print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
        print(f"   ✅ Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Memory test
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"   ✅ GPU Compute: Working")
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ GPU Compute Error: {e}")
    else:
        print(f"   ❌ CUDA Available: False")
        return False
    
    # 3. Package Versions
    print(f"\n📦 PACKAGE VERSIONS:")
    packages = {
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'peft': peft.__version__, 
        'datasets': datasets.__version__,
        'accelerate': accelerate.__version__,
        'bitsandbytes': bitsandbytes.__version__
    }
    
    for pkg, version in packages.items():
        print(f"   ✅ {pkg}: {version}")
    
    # 4. Quick Model Load Test
    print(f"\n🤖 MODEL LOADING TEST:")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test small model loading
        model_name = "microsoft/DialoGPT-small"  # Small test model
        print(f"   Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Test tokenization
        test_text = "Parameter-efficient fine-tuning in scientific domains"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   ✅ Tokenization: {tokens['input_ids'].shape}")
        
        # Test inference
        with torch.no_grad():
            outputs = model(**tokens)
        print(f"   ✅ Model Forward: {outputs.logits.shape}")
        
        print(f"   ✅ Model Loading: Success")
        
    except Exception as e:
        print(f"   ❌ Model Loading Error: {e}")
        return False
    
    # 5. PEFT Test
    print(f"\n🔧 PEFT FUNCTIONALITY TEST:")
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Test LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["c_attn"]  # DialoGPT specific
        )
        
        # Test PEFT model creation
        peft_model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        print(f"   ✅ PEFT Model: Created")
        print(f"   ✅ Trainable Params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    except Exception as e:
        print(f"   ❌ PEFT Error: {e}")
        return False
    
    print(f"\n🎉 ALL TESTS PASSED! Ready to proceed.")
    return True

if __name__ == "__main__":
    success = test_setup()
    if success:
        print(f"\n🚀 Next step: Implement DoRA baseline")
    else:
        print(f"\n❌ Fix errors before proceeding")
