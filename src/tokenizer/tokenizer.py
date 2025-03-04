import re

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# CONSTANTS
SPECIAL_CHARACTERS = r'([,.:;?_!"()\']|--|\s)'


def raw_text_to_list_of_words(text):
    words = re.split(SPECIAL_CHARACTERS, text)
    words = [w.strip() for w in words if w.strip()]
    return words


class VocabularyBuilder:
    """Responsible for building vocabulary based on a list of text sources"""

    def __init__(self):
        self.vocabulary = []

    def _add_to_vocabulary(self, words):
        self.vocabulary += list(words)
        self.vocabulary = list(set(self.vocabulary))

    def _build_vocabulary_for_single_text(self, text):
        words = raw_text_to_list_of_words(text)
        words = sorted(set(words))
        self._add_to_vocabulary(words)
            
    
    def build_vocabulary(self, text_sources):
        for path in text_sources:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                self._build_vocabulary_for_single_text(raw_text)
        
        # add special characters
        self.vocabulary.append("<|unknown|>")
        self.vocabulary.append("<|endoftext|>")
                

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = None
        self.int_to_str = None
        self._tokenize_vocabulary(vocab)

    def _tokenize_vocabulary(self, vocabulary):
        self.int_to_str = {i: s for (i,s) in enumerate(vocabulary)}
        self.str_to_int = {s: i for (i,s) in enumerate(vocabulary)}

    def encode(self, text):
        words = raw_text_to_list_of_words(text)
        words = [w if w in self.str_to_int else "<|unknown|>" 
                 for w in words]
        ids = [self.str_to_int[s] for s in words]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class GPTDatasetV1(Dataset):
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

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    return dataloader

def test():

    # inputs
    text_sources = ["/Users/andylee/Projects/llm-from-scratch/data/the-verdict.txt"]
    path = text_sources[0]
    
    # test vocabulary builder
    vb = VocabularyBuilder()
    vb.build_vocabulary(text_sources)

    # test tokenizer
    # tk = SimpleTokenizerV1(vb.vocabulary)
    tk = tiktoken.get_encoding("gpt2")

    # text1 = "Hello, do you like tea?"
    # text2 = "In the sunlit terraces of the palace."
    # text = " <|endoftext|> ".join((text1, text2))
    # print(text)

    # test dataset loader
    with open(path, "r", encoding='utf-8') as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    second_batch = next(data_iter)
    third_batch = next(data_iter)
    print(first_batch)
    print(second_batch)
    print(third_batch)
    

if __name__ == "__main__":
    test()
