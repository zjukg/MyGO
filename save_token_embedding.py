import torch
from transformers import BertModel

bert_path = "bert-base-uncased"

if __name__ == "__main__":
    bert = BertModel.from_pretrained(bert_path)
    bert_embeddings = bert.get_input_embeddings().weight
    torch.save(bert_embeddings, open("tokens/textual.pth", "wb"))
    print(bert_embeddings.shape)