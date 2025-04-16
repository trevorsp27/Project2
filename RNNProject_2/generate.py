import json
import torch
from models import rnn
from models import lstm
from models import transformer
import sentencepiece as spm

#tokenizer
sp = spm.SentencePieceProcessor()
sp.Load("bpe_tokenizer.model")

#models
vocab_size = sp.get_piece_size()
embed_dim = 256
hidden_dim = 512
num_heads = 8
num_layers = 6

models = {
    "RNN": rnn(vocab_size, embed_dim, hidden_dim),
    "LSTM": lstm(vocab_size, embed_dim, hidden_dim),
    "Transformer": transformer(vocab_size, embed_dim, num_heads, num_layers)
}

#load best checkpoints
for name, model in models.items():
    model.load_state_dict(torch.load(f"{name}_best.pth"))

#samples
prompt = "Which do you prefer? Dogs or cats?"
for name, model in models.items():
    print(f"== {name} ==")
    response = model.prompt(
        prompt,
        sp,
        max_length=50,
        temperature=0.7,
        device="cpu"
    )
    print(f"Response: {response}\n")