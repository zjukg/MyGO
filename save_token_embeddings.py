import torch
from transformers import AutoModel

model_path = "YOUR BERT/RoBERTa/Llama model path"

if __name__ == "__main__":
    model = AutoModel.from_pretrained(model_path)
    token_embeddings = model.get_input_embeddings().weight
    # Change the file name to the corresponding model
    # BERT: textual.pth
    # RoBERTa: textual-roberta.pth
    # Llama: textual-llama.pth
    torch.save(token_embeddings, open("tokens/textual-llama.pth", "wb"))
    print(token_embeddings.shape)