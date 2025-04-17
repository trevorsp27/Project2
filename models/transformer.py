import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024, 
            batch_first=True,
            activation='gelu',
            dropout=0.1 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.register_buffer('mask', self._generate_square_subsequent_mask(max_seq_len))
        
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones(sz, sz)) 
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask
    
    def forward(self, x, temperature=None):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        embed = self.embedding(x) + self.positional_encoding(positions)  
        output = self.transformer(embed, mask=self.mask[:seq_len, :seq_len])
        logits = self.fc(output)
        
        if temperature is not None and not self.training:
            # Use temperature-controlled sampling
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze()
        else:
            return logits
    
    def prompt(self, prompt, tokenizer, max_length=50, temperature=0.7, device="cpu"):
        self.eval()
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                next_token_id = self(input_ids, temperature=temperature).item()
                generated_tokens.append(next_token_id)

                if next_token_id == tokenizer.eos_id():
                    break

                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)

        return tokenizer.decode(generated_tokens)