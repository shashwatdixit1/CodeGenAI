import re

class CustomTokenizer:
    """
    A simple custom tokenizer that assigns unique integer tokens to words in a corpus
    and provides methods to encode and decode text.

    This tokenizer:
    - Sanitizes text by removing special characters and numbers
    - Builds a vocabulary from a training corpus
    - Encodes input strings into lists of token IDs
    - Decodes token ID lists back into human-readable text
    """
    
    def __init__(self):
        """
        Initializes the tokenizer with empty vocabulary and reverse mappings.
        """
        self.vocab = {}
        self.id_to_word = {}

    def sanatize(self, text):
        """
        Removes special characters, digits, and punctuation from the input string.
        """
        return re.sub(r'[^a-zA-Z ]+', '', text)

    def train(self, corpus):
        """
        Trains the tokenizer on the given text corpus.
        Builds a vocabulary of unique words, assigning each a unique token ID.
        """
        words = set()
        for line in corpus:
            clean_line = self.sanatize(line)
            words.update(clean_line.strip().split())

        sorted_words = sorted(words)
        for idx, word in enumerate(sorted_words):
            self.vocab[word] = idx

        for word, idx in self.vocab.items():
            self.id_to_word[idx] = word

    def encode(self, text):
        """
        Converts input text into a list of token IDs using the trained vocabulary.
        """
        clean_text = self.sanatize(text)
        words = clean_text.strip().split()
        encoded_tokens = []
        for word in words:
            if word in self.vocab:
                encoded_tokens.append(self.vocab[word])

        return encoded_tokens

    def decode(self, token_ids):
        """
        Converts a list of token IDs back into a space-separated string of words.
        """
        words = []

        for i in token_ids:
            words.append(self.id_to_word[i])

        decoded_tokens = ' '.join(words)
        return decoded_tokens

training_data = [
    "Hello Shashwat Dixit!",
    "This is tokenizer training data material.",
    "Clean simple words only."
]

tokenizer = CustomTokenizer()
tokenizer.train(training_data)

text = "Hello Shashwat, this is simple!!!"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Original:", text)
print("Encoded Tokens:", encoded)
print("Decoded:", decoded)