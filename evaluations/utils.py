import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# File with several functions that come in handy on multiple occasions.

DS_INCOMPLETE_PATH = '../datasets'
PRETRAINED_VECTORS_PATH = '../pretrained_vectors'
RESULTS_PATH = '../results'


class Split:
    """
    Class used when splitting the training and test set in scikit-learn.

    :attributes:            tfidf - instance of TfidfVectorizer
                            intents_dct, dict - keys: intent labels, values: unique ids
                            new_key_value - keeps track of the newest unique id for intents_dct
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.intents_dct = {}
        self.new_key_value = 0

    def get_X_y(self, lst, fit=True):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.
        Subsequently, it (fits and) transforms the sentences into a matrix of TF-IDF features.

        :returns:           X - feature matrix
                            y - intents encoded using intents_dct as numbers, np.array
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


class Split_BERT:
    """
    Class used when splitting the training and test set in BERT.

    :attributes:            intents_dct, dict - keys: intent labels, values: unique ids
                            new_key_value - keeps track of the newest unique id for intents_dct
    """

    def __init__(self):
        self.intents_dct = {}
        self.new_key_value = 0

    def get_X_y(self, lst):
        """
        Splits a part (contained in lst) of dataset into sentences and intents.
        Subsequently, it (fits and) transforms the sentences into a matrix of TF-IDF features.

        :returns:           X - sentences, list
                            y - intents, list
        """

        X = []
        y = []

        for sent, label in lst:
            if label not in self.intents_dct.keys():
                self.intents_dct[label] = self.new_key_value
                self.new_key_value += 1

            X.append(sent)
            y.append(self.intents_dct[label])

        return X, y


def get_intents_selection(lst, num_samples: int):
    """
    Returns a random selection of intent labels.

    :params:            lst - contains sublists in the following form: [message, label]
                        num_samples, int
    :returns:           selection, (num_samples, ) np.ndarray
    """

    unique_intents = list(set([l[1] for l in lst]))
    selection = np.random.choice(unique_intents, num_samples,
                                 replace=False)  # replace=False doesn't allow elements to repeat

    return selection


def get_filtered_lst(lst, selection):
    """
    Filters a list in order to contain only sublists with intent labels contained in selection.

    :returns:           filtered_lst, list
    """
    filtered_lst = [l for l in lst if l[1] in selection]

    return filtered_lst


def print_results(dataset_size: str, results_dct: dict):
    """Helper print function."""

    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')


def find_best_threshold(val_predictions_labels, oos_label):
    """
    Function used to find the best threshold in oos-threshold.

    :params:            val_predictions_labels - prediction on the validation set, list
                        oos_label - changes when used with scikit-learn or FastText
    :returns:           threshold - best threshold
    """

    # Initialize search for best threshold
    thresholds = np.linspace(0, 1, 101)
    previous_val_accuracy = 0
    threshold = 0

    # Find best threshold
    for idx, tr in enumerate(thresholds):
        val_accuracy_correct = 0
        val_accuracy_out_of = 0

        for pred, label in val_predictions_labels:
            pred_label = pred[0]
            similarity = pred[1]

            if similarity < tr:
                pred_label = oos_label

            if pred_label == label:
                val_accuracy_correct += 1

            val_accuracy_out_of += 1

        val_accuracy = val_accuracy_correct / val_accuracy_out_of

        if val_accuracy < previous_val_accuracy:
            threshold = thresholds[idx - 1]  # best threshold is the previous one
            break

        previous_val_accuracy = val_accuracy
        threshold = tr

    return threshold


def dataset_2_string(lst: list):
    """
    Converts the dataset list into a string that is later converted to file
    in order to be used by FastText's train_supervised() method.

    :params:            lst - contains the dataset, list
    :returns:           ds_str, str
    """

    ds_str = ''

    for sent, label in lst:
        ds_str += f'__label__{label} {sent}\n'

    return ds_str


def get_X_y_fasttext(lst: list):
    """
    Splits the dataset into X and y that are later used in FastText testing.

    :params:            lst - contains the dataset, list
    :returns:           X - contains sentences, list
                        y - contains labels, list
    """

    X, y = [], []

    for sent, label in lst:
        X.append(sent)
        y.append(f'__label__{label}')

    return X, y


def save_results(classifier: str, method: str, dataset_size: str, num_samples: int, repetitions: int,
                 list_results: dict, results: dict):
    """Saves the results of random selection computations into a .txt file."""

    dir_path = os.path.join(RESULTS_PATH, classifier)
    path = os.path.join(dir_path, method + '_results' + '.txt')

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(path, 'a') as f:
        print(
            f'classifier: {classifier}\n'
            f'method: {method}\n'
            f'dataset_size: {dataset_size}\n'
            f'{repetitions} times random select {num_samples} intents\n'
            f'accuracy: {round(results["accuracy"], 1)}\n'
            f'recall: {round(results["recall"], 1)}\n'
            f'far: {round(results["far"], 1)}\n'
            f'frr: {round(results["frr"], 1)}\n'
            f'accuracy list: {list_results["accuracy_lst"]}\n'
            f'recall list: {list_results["recall_lst"]}\n'
            f'far list: {list_results["far_lst"]}\n'
            f'frr list: {list_results["frr_lst"]}\n\n', file=f)
