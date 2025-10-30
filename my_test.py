import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Reasoning-3B",
                                             dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Reasoning-3B")

messages = [
    {"role": "user", "content": "You are analyzing customer support tickets to decide which need escalation.\nTicket 1: 'App crashes when uploading files >50MB.'\nTicket 2: 'Forgot password, can’t log in.'\nTicket 3: 'Billing page missing enterprise pricing.'\nClassify each ticket as Critical, Medium, or Low and explain your reasoning.\n"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

outputs = model.generate(**tokenizer(prompts, return_tensors="pt").to(model.device), do_sample=True, temperature=0.6, max_new_tokens=4096)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
