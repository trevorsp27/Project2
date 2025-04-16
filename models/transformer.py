import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.register_buffer('mask', self._generate_square_subsequent_mask(max_seq_len))
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, temperature=None):
        seq_len = x.size(1)
        embed = self.embedding(x)
        embed += self.positional_encoding[:, :seq_len, :]
        output = self.transformer(embed, mask=self.mask[:seq_len, :seq_len])
        logits = self.fc(output)
        
        if temperature is not None and not self.training:
            #apply temperature-controlled sampling during generation
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze()
        else:
            #return logits during training
            return logits
    
    def prompt(self, prompt, tokenizer, max_length=50, temperature=1.0, device="cpu"):
        self.eval()
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                generated_tokens.append(next_token_id)

                if next_token_id == tokenizer.eos_id():
                    break

                # Append the new token and ensure it's on the same device
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)

        return tokenizer.decode(generated_tokens)