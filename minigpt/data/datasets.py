import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = tokenizer

        # build dataset
        self.input_ids = []
        self.target_ids = []
        self._build_dataset(text)

    def _build_dataset(self, text):

        context_length = self.max_length
        token_ids = self.tokenizer.encode(text)
        final_token_index = len(token_ids) - self.max_length

        for i in range(0, final_token_index, self.stride):
            input_chunk = token_ids[i: i+context_length]
            target_chunk = token_ids[i+1 : i+1+context_length]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
