import torch
from torch.utils.data import IterableDataset

class TextIterableDataset(IterableDataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

    def __iter__(self):
        f = open(self.file_path, "r", encoding="utf-8")
        for line in f:
            line = line.strip()
            if len(line) > 0:
                tokenized_line = self.tokenizer.encode(line)
                for i in range(0, len(tokenized_line), self.block_size):
                    yield torch.tensor(tokenized_line[i:i+self.block_size], dtype=torch.long)
