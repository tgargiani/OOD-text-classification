import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Utils:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.intents_dct = {}
        self.new_key_value = 0

    def get_X_y(self, lst, fit=True):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.
        Subsequently, it (fits and) transforms the sentences into a matrix of TF-IDF features.
        Returns:
            X - feature matrix
            y - np.array of intents encoded using intents_dct as numbers
        """

        sentences = []
        intents = []

        for sent, label in lst:
            if label not in self.intents_dct.keys():
                self.intents_dct[label] = self.new_key_value
                self.new_key_value += 1

            sentences.append(sent)
            intents.append(self.intents_dct[label])

        if fit:
            X = self.tfidf.fit_transform(sentences)
        else:
            X = self.tfidf.transform(sentences)

        y = np.asarray(intents)

        return X, y

    def get_intents_selection(self, lst, num_samples):
        unique_intents = list(set([l[1] for l in lst]))
        selection = np.random.choice(unique_intents, num_samples,
                                     replace=False)  # replace=False doesn't allow elements to repeat

        return selection

    def get_filtered_lst(self, lst, selection):
        filtered_lst = [l for l in lst if l[1] in selection]

        return filtered_lst
