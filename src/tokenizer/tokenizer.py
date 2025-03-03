import re

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
        ids = [self.str_to_int[s] for s in words]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def test():

    # test vocabulary builder
    text_sources = ["/Users/andylee/Projects/llm-from-scratch/data/the-verdict.txt"]
    vb = VocabularyBuilder()
    vb.build_vocabulary(text_sources)

    # test tokenizer
    tk = SimpleTokenizerV1(vb.vocabulary)
    path = "/Users/andylee/Projects/llm-from-scratch/data/the-verdict.txt"
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        raw_text = raw_text[:400]
        print("raw_text : ", raw_text)
        enocded = tk.encode(raw_text)
        print("tokenized: ", tk.encode(raw_text))
        print("characters_decoded " , tk.decode(enocded))


if __name__ == "__main__":
    test()
