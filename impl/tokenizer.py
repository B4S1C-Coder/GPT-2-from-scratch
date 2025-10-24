import re

class SimpleTokenizerV1:
    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }
        self.eot = "<|endoftext|>"
        self.unk = "<|unk|>"

    def encode(self, text: str) -> list[int]:
        preprocessed: list[str] = [
            item.strip() for item in
            re.split(r'([,.:;?_!"()\']|--|\s)', text)
            if item.strip()
        ]

        ids = [
            self.str_to_int[s] if s in self.str_to_int
            else self.str_to_int[self.unk]
            for s in preprocessed
        ]

        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

def main():
    with open('datasets_/the-verdict.txt', 'r', encoding="utf-8") as f:
        raw_text = f.read()

    print("Total characters in raw text:", len(raw_text))

    preprocessed = [
        item.strip() for item in 
        re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        if item.strip()
    ]

    print("Total tokens in preprocessed:", len(preprocessed))

    # Creating Tokens
    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)

    print("Vocab Size:", vocab_size)

    # This process is essentially encoding: text -> numbers
    vocab = { token:integer for integer,token in enumerate(all_words) }

    tokenizer = SimpleTokenizerV1(vocab)
    
    text = """"It's the last he painted, you know,"
               Mrs. Gisburn said with pardonable pride."""

    ids = tokenizer.encode(text)
    print(ids)

    decoded = tokenizer.decode(ids)
    print(decoded)

    text = "Hello, world! How are you doing today? Arrt is no longer used in the present times."
    
    ids = tokenizer.encode(text)
    print(ids)

    decoded = tokenizer.decode(ids)
    print(decoded)

if __name__ == "__main__":
    main()
