#!/usr/bin/env python3
"""
Standalone test for AI21 Jamba Reasoning 3B model
Tests the model directly without the API
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_jamba_model(prompt: str):
    """Test Jamba model directly"""
    
    model_name = "ai21labs/AI21-Jamba-Reasoning-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print(f"🧪 Testing AI21 Jamba Reasoning 3B Model")
    print(f"📍 Device: {device}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80)
    
    try:
        # Load tokenizer
        print("\n📥 Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False  # Use slow tokenizer to avoid format issues
            )
        except Exception as e:
            print(f"⚠️  Failed with slow tokenizer, trying with fast tokenizer and legacy=True...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
                legacy=False
            )
        
        # Set special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        
        # Load model
        print(f"\n📥 Loading model from {model_name}...")
        print("⚠️  This may take a few minutes...")
        
        if device == "cuda":
            # Load with optimizations for GPU
            # Use naive Mamba implementation (slower but doesn't require kernels)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.use_mamba_kernels = False  # Disable fast kernels, use naive implementation
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16,  # Use FP16 for efficiency
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"✅ Model loaded on GPU with FP16 (using naive Mamba implementation)")
        else:
            # Load for CPU
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.use_mamba_kernels = False
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            print(f"✅ Model loaded on CPU (using naive Mamba implementation)")
        
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        
        if device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"💾 GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        # Generate text
        print("\n" + "="*80)
        print(f"📝 Prompt: {prompt}")
        print("="*80)
        print("\n🔄 Generating response...\n")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        if full_output.startswith(prompt):
            generated_text = full_output[len(prompt):].strip()
        else:
            generated_text = full_output.strip()
        
        print("="*80)
        print("🎯 GENERATED RESPONSE:")
        print("="*80)
        print(generated_text)
        print("="*80)
        
        # Show token stats
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(outputs[0])
        generated_tokens = output_tokens - input_tokens
        
        print(f"\n📊 Token Statistics:")
        print(f"   Input tokens: {input_tokens}")
        print(f"   Generated tokens: {generated_tokens}")
        print(f"   Total tokens: {output_tokens}")
        
        if device == "cuda":
            final_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"\n💾 Final GPU Memory Allocated: {final_allocated:.2f} GB")
        
        print("\n✅ Test completed successfully!")
        return generated_text
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        logger.error(f"Jamba test failed: {e}", exc_info=True)
        return None
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 GPU cache cleared")

def main():
    # Default prompt if none provided
    default_prompt = "What is a barrel shifter in digital circuits? Explain its working principle."
    
    # Check if prompt provided as command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = default_prompt
        print(f"ℹ️  No prompt provided, using default prompt")
        print(f"ℹ️  Usage: python test_jamba_standalone.py \"Your prompt here\"\n")
    
    # Run test
    test_jamba_model(prompt)

if __name__ == "__main__":
    main()
