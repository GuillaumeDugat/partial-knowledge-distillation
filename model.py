from transformers import AutoTokenizer, TFAutoModelForCausalLM
import torch

# with transformers
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = TFAutoModelForCausalLM.from_pretrained(
    "distilgpt2", pad_token_id=tokenizer.eos_token_id
)

# with torch
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2') 
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'distilgpt2') 
