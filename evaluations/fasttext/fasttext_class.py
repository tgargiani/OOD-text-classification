import fasttext
import numpy as np
from keras_preprocessing.text import text_to_word_sequence

FASTTEXT = '/Users/tommaso.gargiani/Documents/Alquist/cc.en.300.bin'

class FastText(object):
    def __init__(self, padding=40):
        self.e = fasttext.load_model(FASTTEXT)
        self.padding = padding

    def texts_to_sequences(self, texts):
        sentence_features = np.zeros((len(texts), self.padding, self.e.get_dimension()))
        for j, text in enumerate(texts):
            words = text_to_word_sequence(text)
            for i, word in enumerate(words):
                if i >= self.padding:
                    break
                sentence_features[j, i, :] = self.e.get_word_vector(word)
        return sentence_features

    def tokens_to_embeddings(self, token_list):
        sentence_features = np.zeros((self.padding, self.e.get_dimension()))
        for i, word in enumerate(token_list):
            if i >= self.padding:
                break
            sentence_features[i, :] = self.e.get_word_vector(word)
        return sentence_features
