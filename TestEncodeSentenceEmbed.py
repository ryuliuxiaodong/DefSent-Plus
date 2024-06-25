from defsentplus import DefSentPlus



if __name__ == "__main__":
    sentences = ['A woman is reading.', 'A man is playing a guitar.', 'He plays guitar.', 'A woman is making a photo.']
    # sentences = sentences * 100

    # Available backbone model name is bert-base-uncased, bert-large-uncased, roberta-base, or roberta-large.
    encoder = DefSentPlus("RyuKT/DefSentPlus-bert-base-uncased", backbone_model_name="bert-base-uncased", device="cuda")

    # Available pooling is "cls", "mean", or "prompt".
    # Default batch size is 16.
    embeds = encoder.encode(sentences=sentences, pooling="prompt", batch_size=16)

    print(embeds)
    print(embeds.shape)