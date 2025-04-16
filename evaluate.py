import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
import json
import matplotlib.pyplot as plt

def calculate_perplexity(model, data_loader, pad_token_id, device="cpu"):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return torch.exp(torch.tensor(avg_loss)).item()

from nltk.translate.bleu_score import sentence_bleu
from multiprocessing import Pool

def compute_bleu_for_example(reference, candidate):
    """
    Helper function to compute BLEU score for a single example.
    """
    return sentence_bleu([reference], candidate)

def calculate_bleu_parallel(references, candidates):
    with Pool() as pool:
        scores = pool.starmap(compute_bleu_for_example, zip(references, candidates))
    return sum(scores) / len(scores)

def calculate_bleu(model, test_examples, tokenizer, device="cpu"):
    print("Starting BLEU calculation...")
    model.eval()
    references = []
    candidates = []

    for i, example in enumerate(test_examples):
        if i % 100 == 0:
            print(f"Processing example {i}/{len(test_examples)}")
        
        prompt = example["prompt"]
        ground_truth_tokens = example["completion"]  # Already tokenized

        # Generate text using the model
        generated = model.prompt(
            prompt,
            tokenizer,
            max_length=512,
            temperature=0.7,
            device=device
        )
        
        # Tokenize the generated text
        candidate_tokens = tokenizer.encode(generated, out_type=str)

        # Debugging: Print the first example's reference and candidate
        if i == 0:
            print("Example Reference:", [ground_truth_tokens])
            print("Example Candidate:", candidate_tokens)

        references.append([ground_truth_tokens])  # NLTK expects list of references
        candidates.append(candidate_tokens)

    print("Calculating BLEU score...")
    bleu_score = calculate_bleu_parallel(references, candidates)
    print("BLEU score calculated.")
    return bleu_score
