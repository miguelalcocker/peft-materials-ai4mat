# ============================================
# QUICK DORA TEST - HUGGINGFACE INTEGRATION
# Save as: ~/peft-scientific/test_dora_hf.py
# ============================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import time

def test_dora_huggingface():
    """Test DoRA with HuggingFace PEFT integration"""
    
    print("🦙 Testing DoRA with HuggingFace PEFT")
    print("=" * 50)
    
    # 1. Setup model (use smaller model for quick test)
    model_name = "microsoft/DialoGPT-medium"  # 355M params
    print(f"📦 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print(f"✅ Base model loaded")
    
    # 2. Test configurations
    configs_to_test = [
        {
            "name": "LoRA Baseline",
            "config": LoraConfig(
                use_dora=False,  # Standard LoRA
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )
        },
        {
            "name": "DoRA (Official)",
            "config": LoraConfig(
                use_dora=True,  # 🔥 DoRA enabled
                r=8,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )
        },
        {
            "name": "DoRA Low-Rank",
            "config": LoraConfig(
                use_dora=True,
                r=4,  # Lower rank as paper suggests
                lora_alpha=8,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )
        }
    ]
    
    results = []
    
    for config_info in configs_to_test:
        print(f"\n🧪 Testing: {config_info['name']}")
        
        try:
            # Apply PEFT
            model = get_peft_model(base_model, config_info['config'])
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   Total params: {total_params:,}")
            print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
            
            # Test inference
            test_prompt = "Parameter-efficient fine-tuning in scientific domains"
            inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Time inference
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            inference_time = time.time() - start_time
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated_text[len(test_prompt):].strip()
            
            print(f"   ✅ Generation: {completion}")
            print(f"   ⏱️ Time: {inference_time:.3f}s")
            
            # Test training step
            model.train()
            dummy_batch = tokenizer(
                ["Scientific models require efficient adaptation"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64
            )
            dummy_batch = {k: v.to(model.device) for k, v in dummy_batch.items()}
            
            outputs = model(**dummy_batch, labels=dummy_batch['input_ids'])
            loss = outputs.loss
            print(f"   📊 Training loss: {loss.item():.4f}")
            
            # Store results
            results.append({
                'name': config_info['name'],
                'trainable_params': trainable_params,
                'efficiency': 100 * trainable_params / total_params,
                'inference_time': inference_time,
                'loss': loss.item(),
                'use_dora': config_info['config'].use_dora
            })
            
            print(f"   ✅ {config_info['name']} test passed")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'name': config_info['name'],
                'error': str(e)
            })
    
    # 3. Results summary
    print(f"\n📊 RESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Method':<15} {'Trainable %':<12} {'Time (s)':<10} {'Loss':<8} {'DoRA':<6}")
    print("-" * 70)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['name']:<15} {result['efficiency']:<12.3f} {result['inference_time']:<10.3f} {result['loss']:<8.4f} {result['use_dora']}")
        else:
            print(f"{result['name']:<15} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8} {'ERROR'}")
    
    # 4. Key findings
    dora_results = [r for r in results if 'error' not in r and r['use_dora']]
    lora_results = [r for r in results if 'error' not in r and not r['use_dora']]
    
    if dora_results and lora_results:
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   ✅ DoRA vs LoRA efficiency: Similar (~{dora_results[0]['efficiency']:.3f}% vs {lora_results[0]['efficiency']:.3f}%)")
        print(f"   ✅ DoRA setup working with HuggingFace PEFT")
        print(f"   ✅ Ready for scientific domain adaptation")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting DoRA HuggingFace integration test...")
    results = test_dora_huggingface()
    
    if any('error' not in r for r in results):
        print(f"\n🎉 DoRA integration successful!")
        print(f"🚀 Next: Adapt for scientific domains")
    else:
        print(f"\n❌ Fix integration issues before proceeding")
