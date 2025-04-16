import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, temperature=None):
        embed = self.embedding(x)
        output, _ = self.lstm(embed)
        logits = self.fc(output)
        
        if temperature is not None and not self.training:
            #apply temperature-controlled sampling during generation
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze()
        else:
            #return logits during training
            return logits
    
    def prompt(self, prompt, tokenizer, max_length=50, temperature=0.7, device="cpu"):
        self.eval()
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                generated_tokens.append(next_token_id)

                #stop if EOS token is generated
                if next_token_id == tokenizer.eos_id():
                    break

                #append the predicted token
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)

        return tokenizer.decode(generated_tokens)