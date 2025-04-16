import json
import sentencepiece as spm

#combine all training text
with open("train.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]
all_text = [f"{ex['prompt']} {ex['completion']}" for ex in lines]

#save to temporary file for SentencePiece
with open("train.txt", "w") as f:
    f.write("\n".join(all_text))

#train BPE tokenizer
spm.SentencePieceTrainer.train(
    input="train.txt",
    model_prefix="bpe_tokenizer",
    vocab_size=10000,
    pad_id=0,  #explicitly set pad token
    unk_id=1,  #unknown token
    bos_id=2,  #begin-of-sequence
    eos_id=3   #end-of-sequence
)