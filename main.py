from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional

# Initialize the FastAPI app
app = FastAPI()

class SmolLMModel:
    """SmolLM2-1.7B-Instruct model implementation - lightweight SLM for API use"""
    def __init__(self, model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading {model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True
            ).to(self.device)
        print(f"SmolLM2 model loaded successfully on {self.device}")
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            input_length = inputs['input_ids'].shape[1]
            max_new_tokens = max(10, max_length - input_length)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            return response
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")

class DeepSeekModel:
    """DeepSeek-R1-Distill-Qwen-7B model implementation for CVDP benchmark"""
    def __init__(self, model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model_name = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading {model}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer for {model}: {e}")
            fallback_model = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            print(f"Attempting fallback to {fallback_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            self.model_name = fallback_model
            model = fallback_model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True
            ).to(self.device)
        print(f"DeepSeek model loaded successfully on {self.device}")
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            input_length = inputs['input_ids'].shape[1]
            max_new_tokens = max(10, max_length - input_length)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            return response
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")

# IMPORTANT: Load the model once when the application starts up.
# This prevents reloading the heavy model on every single API call.
generator = SmolLMModel()

print("Loading SmolLM model...")
smollm_generator = SmolLMModel()
print("SmolLM model loaded successfully!")

print("Loading DeepSeek model...")
deepseek_generator = DeepSeekModel()
print("DeepSeek model loaded successfully!")


# Define the request data structure using Pydantic
class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 10  # Increased default from 50 to 100
    model: Optional[str] = "smollm"  # "smollm" or "deepseek"

@app.get("/")
def read_root():
    return {
        "status": "SLM API is running",
        "models": [
            {"name": smollm_generator.model_name, "type": "SmolLM2-1.7B-Instruct", "device": smollm_generator.device},
            {"name": deepseek_generator.model_name, "type": "DeepSeek-R1-Distill-Qwen-7B", "device": deepseek_generator.device}
        ]
    }

@app.get("/model_info")
def get_model_info():
    return {
        "smollm": {
            "model_name": smollm_generator.model_name,
            "device": smollm_generator.device,
            "model_type": "SmolLM2-1.7B-Instruct"
        },
        "deepseek": {
            "model_name": deepseek_generator.model_name,
            "device": deepseek_generator.device,
            "model_type": "DeepSeek-R1-Distill-Qwen-7B"
        }
    }


# Define the API endpoint
@app.post("/generate")
def generate_text(request: PromptRequest):
    """
    Accepts a prompt and returns the model's generated text.
    Use 'model' field in request to select between 'smollm' and 'deepseek'.
    """
    try:
        if request.model == "deepseek":
            generated_text = deepseek_generator.generate_text(request.prompt, max_length=request.max_length)
        else:
            generated_text = smollm_generator.generate_text(request.prompt, max_length=request.max_length)
        return {"response": generated_text, "model": request.model}
    except Exception as e:
        return {"error": str(e), "model": request.model}