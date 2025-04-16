import json
import torch
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collation function to pad sequences in a batch.
    Args:
        batch: List of tuples, where each tuple contains (input_ids, labels).
    Returns:
        A dictionary with padded "input_ids" and "labels".
    """
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Use 0 for input padding
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)     # Use -100 for label padding

    return {"input_ids": input_ids, "labels": labels}

import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()
        self.max_len = max_len
        self.examples = []

        with open(jsonl_file, "r") as f:
            for line in f:
                example = json.loads(line)
                if "prompt" in example and "completion" in example:
                    text = f"{example['prompt']} {example['completion']}"
                    tokens = tokenizer.encode(text)
                    # Validate tokens
                    if any(token < 0 or token >= self.vocab_size for token in tokens):
                        print(f"Skipping invalid tokens: {tokens}")
                        continue
                    self.examples.append(tokens[:self.max_len])

    def __len__(self):
        return len(self.examples)  # Fix: Use "examples" instead of "tokenized"

    def __getitem__(self, idx):
        tokens = self.examples[idx]  # Fix: Use "examples" instead of "tokenized"
        x = tokens[:-1]
        y = tokens[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)