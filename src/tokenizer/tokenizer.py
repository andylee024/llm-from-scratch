import re

class VocabularyBuilder:
    """Responsible for building vocabulary based on a list of text sources"""

    def __init__(self):
        self.special_characters = r'([,.:;?_!"()\']|--|\s)'
        self.vocabulary = []

    def _add_to_vocabulary(self, words):
        self.vocabulary += list(words)
        self.vocabulary = list(set(self.vocabulary))

    def _build_vocabulary_for_single_text(self, text):
        words = re.split(self.special_characters, text)
        words = [w.strip() for w in words if w.strip()]
        words = sorted(set(words))
        self._add_to_vocabulary(words)
            
    
    def build_vocabulary(self, text_sources):
        for path in text_sources:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                self._build_vocabulary_for_single_text(raw_text)
                

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def test():
    text_sources = ["/Users/andylee/Projects/llm-from-scratch/data/the-verdict.txt"]
    vb = VocabularyBuilder()
    vb.build_vocabulary(text_sources)

    print(sorted(vb.vocabulary))

if __name__ == "__main__":
    test()
