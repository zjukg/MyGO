import torch
from transformers import AutoModel

model_path = "/home/zhangyichi/Huawei2023/llm-zoo/llama-7b"

if __name__ == "__main__":
    model = AutoModel.from_pretrained(model_path)
    token_embeddings = model.get_input_embeddings().weight
    torch.save(token_embeddings, open("tokens/textual-llama.pth", "wb"))
    print(token_embeddings.shape)