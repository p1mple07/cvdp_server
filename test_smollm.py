#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_smollm():
    """Test SmolLM2 model directly"""
    
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Testing SmolLM2 on {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        
        # Load model
        print("Loading model...")
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(device)
        
        model.eval()
        print(f"Model loaded successfully on {device}")
        
        # Test simple generation
        test_prompts = [
            "Hello, how are you?",
            "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nExplain what a barrel shifter is in simple terms.<|im_end|>\n<|im_start|>assistant\n"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {repr(prompt[:50])}...")
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
                input_length = inputs['input_ids'].shape[1]
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=100,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        use_cache=True
                    )
                
                response_tokens = outputs[0][input_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                
                print(f"Response ({len(response_tokens)} tokens): {response}")
                
            except Exception as e:
                print(f"Generation failed: {e}")
                
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smollm()