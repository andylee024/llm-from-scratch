import numpy as np
import os
import torch

from torch.utils.data import Dataset

class BinaryDataset(Dataset):
    """Implementation of dataset object that handles binary files"""

    def __init__(self, binary_file):
        self.binary_file = binary_file

    def __len__(self):
        data = np.memmap(self.binary_file, dtype=np.uint16, mode='r')
        return len(data)

    def __getitem__(self, idx):
        data = np.memmap(self.binary_file, dtype=np.uint16, mode='r')
        # Convert scalar to array or handle array properly
        if np.isscalar(data[idx]):
            return torch.tensor(int(data[idx]), dtype=torch.int64)
        else:
            return torch.from_numpy(data[idx].astype(np.int64))
    
    def get_random_batch(self, batch_size, block_size, device_type, device='cuda'):
        """Get random batch"""
        data = np.memmap(self.binary_file, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


class TextDataset(Dataset):
    """Implementation of dataset object that handles text files"""
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

def test():
    """Test the BinaryDataset class with Shakespeare data."""
    
    # Test BinaryDataset
    print("Testing BinaryDataset...")
    shakespeare_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'shakespeare')
    train_file = os.path.join(shakespeare_dir, 'train.bin')
    
    # Create dataset
    dataset = BinaryDataset(train_file)
    print(f"Dataset length: {len(dataset)}")
    
    # Test __getitem__
    sample = dataset[1]
    print(f"Sample token: {sample}")
    
    # Test get_random_batch
    batch_size = 4
    block_size = 64
    x, y = dataset.get_random_batch(batch_size, block_size, device_type='cpu', device='cpu')
    print(f"Random batch shapes: x={x.shape}, y={y.shape}")
    print(f"First sequence in batch: {x[0][:10]}...")
    print(f"First target in batch: {y[0][:10]}...")
    
    # Verify that y is x shifted by 1
    for i in range(batch_size):
        assert torch.equal(x[i, 1:], y[i, :-1]), "Target should be input shifted by 1"
    
    print("BinaryDataset test completed successfully!")


if __name__ == "__main__":
    test()


