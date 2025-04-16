import json
import torch
from torch.utils.data import DataLoader
import sentencepiece as spm
from data_utils import TextDataset
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from train import train_model
from evaluate import calculate_perplexity, calculate_bleu
from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from data_utils import TextDataset, collate_fn
from torch.utils.data import Subset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load("bpe_tokenizer.model")
    print(f"Tokenizer vocabulary size: {sp.vocab_size()}")  # Should print 10000
    assert sp.vocab_size() == 10000, "Tokenizer vocabulary size is not 10000!"
    vocab_size = sp.vocab_size()
    pad_token_id = sp.pad_id()
    
    # Initialize the dataset
    train_dataset = TextDataset("train.jsonl", sp, max_len=512)
    test_dataset = TextDataset("test.jsonl", sp, max_len=512)

    # Initialize the DataLoader with collate_fn
   # DataLoader initialization in main.py
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn  # Ensure this is included
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=collate_fn  # Ensure this is included
    )
    
    # Models
    models = {
        "RNN": RNNModel(vocab_size, embed_dim=256, hidden_dim=512),
        "LSTM": LSTMModel(vocab_size, embed_dim=256, hidden_dim=512),
        "Transformer": TransformerModel(
            vocab_size=vocab_size,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            max_seq_len=512
        )
    }
    
    # Training and evaluation loop
    for name, model in models.items():
        print(f"\n{'='*20} Training {name} {'='*20}")
        model.to(device)  # Move model to GPU before training
        
        # Train the model
        train_model(
            model,
            train_loader,
            test_loader,
            pad_token_id=pad_token_id,
            epochs=30,
            patience=5,
            device=device
        )
        
        # Load best checkpoint and move to GPU
        model.load_state_dict(torch.load(f"{model.__class__.__name__}_best.pth", map_location=device))
        model.to(device)  # Ensure the model is on the GPU
        
        # Generate text for the required prompt
        print("Prompt: Which do you prefer? Dogs or cats?")
        response = model.prompt(
            prompt="Which do you prefer? Dogs or cats?",
            tokenizer=sp,
            max_length=50,
            temperature=0.7,
            device=device  # Pass the device to the prompt method
        )
        print(f"Response: {response}")

        # Generate text for a custom prompt
        print("Prompt: Once upon a time...")
        response = model.prompt(
            "Once upon a time...",
            sp,
            max_length=100,
            temperature=0.8,
            device=device
        )
        print(f"Response: {response}\n")

        # Calculate Perplexity
        ppl = calculate_perplexity(model, test_loader, pad_token_id, device=device)
        print(f"Perplexity: {ppl:.2f}")



        # Calculate BLEU Score
        # Read and process the dataset in one pass
        with open("test.jsonl", "r") as f:
            test_examples = []
            print("Reading test.jsonl...")
            for i, line in enumerate(f):
                try:
                    example = json.loads(line)
                    assert "prompt" in example and "completion" in example
                    # Tokenize the completion field
                    tokenized_completion = sp.encode(example["completion"], out_type=str)
                    test_examples = torch.load("tokenized_test.pt")
                    test_examples = [100]
                    if i % 100 == 0:
                        print(f"Processed {i} examples...")
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}. Error: {e}")
            print(f"Finished reading {len(test_examples)} examples.")    




        
        #TESTING CALCBLEU
        test_examples = [
        {"prompt": "Which do you prefer? Dogs or cats?", "completion": "I prefer dogs."},
        {"prompt": "Once upon a time...", "completion": "there was a brave knight."}
        ]
        # Tokenize the dataset
        for example in test_examples:
            example["completion"] = sp.encode(example["completion"], out_type=str)
        # Test the calculate_bleu function
        print("Testing calculate_bleu function...")
        bleu = calculate_bleu(model, test_examples, sp, device)
        print(f"BLEU Score: {bleu:.4f}")






        print(f"Number of test examples: {len(test_examples)}")
        if len(test_examples) == 0:
            raise ValueError("test_examples is empty. Please check the dataset loading process.")
        print("About to calculate BLEU score...")
        bleu = calculate_bleu(model, test_examples, sp, device)
        print(f"BLEU Score: {bleu:.4f}\n")

if __name__ == "__main__":
    freeze_support()
    main()