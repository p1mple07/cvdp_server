from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class NemotronModel:
    """NVIDIA Nemotron model implementation for CVDP benchmark"""
    
    def __init__(self, model: str = "nvidia/Nemotron-Mini-4B-Instruct"):
        self.model_name = model
        self.requires_evaluation = True
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Check GPU memory and adjust model if needed
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory available: {gpu_memory_gb:.1f} GB")
            
            # If trying to load a large model on limited GPU, switch to Mini
            if "340B" in model and gpu_memory_gb < 80:
                logger.info(f"Switching to Nemotron-Mini due to GPU memory constraints ({gpu_memory_gb:.1f} GB < 80 GB)")
                model = "nvidia/Nemotron-Mini-4B-Instruct"
                self.model_name = model
        
        # Load tokenizer and model
        logger.info(f"Loading {model}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model}: {e}")
            # Try fallback to a compatible model or raise error
            raise ValueError(f"Cannot load tokenizer for {model}. This may require a newer version of transformers library.")
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        try:
            if torch.cuda.is_available():
                # For GPU - use more conservative settings to avoid CUDA kernel issues
                logger.info("Using conservative GPU settings to avoid CUDA kernel issues...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for better P100 compatibility
                    trust_remote_code=True,
                    load_in_8bit=False,  # Disable quantization to avoid cutlass kernel issues
                    load_in_4bit=False,
                    device_map=None,  # Load on single device to avoid multi-GPU issues
                    low_cpu_mem_usage=True
                ).to(self.device)
            else:
                # For CPU (not recommended for large Nemotron models)
                logger.info("Warning: Running large Nemotron models on CPU may be very slow")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model, 
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    load_in_8bit=False
                ).to(self.device)
        except Exception as e:
            logger.error(f"Error loading model {model}: {e}")
            # Check if it's an unsupported architecture error
            if "model type" in str(e) and "nemotron" in str(e).lower():
                raise ValueError(f"Nemotron model architecture not supported in current transformers version. Please upgrade transformers: pip install --upgrade transformers")
            else:
                raise e
        
        # Set model to eval mode
        self.model.eval()
        logger.info(f"Nemotron model loaded successfully on {self.device}")
        logger.info(f"Context window: 4096 tokens (input + output combined)")
        logger.info(f"Maximum output tokens: 4096 tokens")
    
    def generate_text(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate response using NVIDIA Nemotron model
        """
        try:
            # Format prompt for Nemotron models using chat template
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant specialized in code generation and technical tasks."},
                {"role": "user", "content": prompt}
            ]
            
            # Try to apply chat template, fallback to direct formatting if not available
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback formatting for Nemotron
                formatted_prompt = f"<|system|>\nYou are a helpful AI assistant specialized in code generation and technical tasks.\n<|user|>\n{prompt}\n<|assistant|>\n"

            # Tokenize with maximum context length for Nemotron (4K tokens)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(self.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Calculate max_new_tokens - Nemotron supports up to 4K total context
            max_new_tokens = min(max(100, max_length - input_length), 4096)

            # Generate response with parameters optimized for Nemotron
            with torch.no_grad():
                try:
                    # First try with conservative settings
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        temperature=max(0.1, temperature),  # Ensure temperature is at least 0.1
                        top_p=top_p,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.05,
                        use_cache=True,
                        # Additional parameters to avoid CUDA kernel issues
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict_in_generate=False
                    )
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "cutlass" in error_msg or "kernel" in error_msg or "out of memory" in error_msg:
                        logger.error(f"CUDA error encountered: {e}")
                        logger.info("Retrying with simplified generation...")
                        
                        # Try with simpler generation parameters first
                        try:
                            outputs = self.model.generate(
                                inputs['input_ids'],
                                max_new_tokens=min(500, max_new_tokens),  # Reduce tokens
                                temperature=0.1,
                                do_sample=False,  # Disable sampling to avoid complex kernels
                                pad_token_id=self.tokenizer.eos_token_id,
                                use_cache=False,  # Disable cache to avoid memory issues
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict_in_generate=False
                            )
                        except RuntimeError as e2:
                            logger.error(f"GPU generation failed again: {e2}")
                            logger.info("Falling back to CPU generation...")
                            # Move to CPU and retry
                            self.model = self.model.to("cpu")
                            inputs = {k: v.to("cpu") for k, v in inputs.items()}
                            outputs = self.model.generate(
                                inputs['input_ids'],
                                max_new_tokens=min(500, max_new_tokens),  # Reduce tokens for CPU
                                temperature=0.1,
                                do_sample=False,  # Disable sampling on CPU for stability
                                pad_token_id=self.tokenizer.eos_token_id,
                                use_cache=False  # Disable cache on CPU
                            )
                            # Move back to GPU for next iteration
                            if torch.cuda.is_available():
                                self.model = self.model.to(self.device)
                    else:
                        raise e

            # Decode only new tokens
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # Clean up response - remove common chat template artifacts
            response = response.replace("<|assistant|>", "").replace("<|end|>", "").replace("<|system|>", "").replace("<|user|>", "").strip()
            if response.startswith("assistant"):
                response = response[len("assistant"):].strip()

            logger.info(f"Nemotron generated {len(response_tokens)} tokens")
            
            return response if response else "Unable to generate response."
            
        except Exception as e:
            logger.error(f"Error generating Nemotron response: {e}")
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def __str__(self):
        return f"NemotronModel({self.model_name})"
    
    def __del__(self):
        # Clean up GPU memory when object is destroyed
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
