from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("vicgalle/gpt2-alpaca-gpt4")
    # tokenizer does not have a padding token by default
    # --> setting it to be the same as the end of sentence token
    tokenizer.pad_token = tokenizer.eos_token_id
    return tokenizer


def get_pretrained_gpt2():
    model = GPT2LMHeadModel.from_pretrained("vicgalle/gpt2-alpaca-gpt4")
    return model


def get_pretrained_distilgpt2():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return model


def get_untrained_distilgpt2(config):
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.n_layer = config["distilled_model"]["nb_layers"]
    model = GPT2LMHeadModel(gpt2_config)
    return model


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    model = get_pretrained_distilgpt2()
    inputs = tokenizer("Hi, this is a test sentence", return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    print("loss", outputs.loss)
    print("logits", outputs.logits)
