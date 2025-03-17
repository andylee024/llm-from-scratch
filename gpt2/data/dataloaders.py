import tiktoken
from torch.utils.data import DataLoader
from data.datasets import GPTDataset

def create_dataloader(txt, 
                      batch_size=4, 
                      max_length=256, 
                      stride=128, 
                      shuffle=True, 
                      drop_last=True, 
                      num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader