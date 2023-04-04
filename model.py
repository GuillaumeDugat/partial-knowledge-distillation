from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

def get_tokenizer(): 
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_pretrained_gpt2():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model


def get_pretrained_distilgpt2():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return model

def get_untrained_distilgpt2():
    config = GPT2Config.from_pretrained("gpt2")
    config.n_layer = 6
    model = GPT2LMHeadModel(config)
    return model



if __name__ == "__main__" : 
    tokenizer = get_tokenizer()
    model = get_pretrained_distilgpt2()
    inputs = tokenizer("Hi, this is a test sentence", return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    print("loss", outputs.loss)
    print("logits", outputs.logits)