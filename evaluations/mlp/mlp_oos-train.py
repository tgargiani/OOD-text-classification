from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import json, os
import numpy as np


def get_X_y(lst, fit=True):
    """
    Splits a part (contained in lst) of dataset into sentences and intents.
    Subsequently, it (fits and) transforms the sentences into a matrix of TF-IDF features.
    Returns:
        X - feature matrix
        y - np.array of intents encoded using intents_dct as numbers
    """

    global tfidf
    global intents_dct
    global new_key_value

    sentences = []
    intents = []

    for sent, label in lst:
        if label not in intents_dct.keys():
            intents_dct[label] = new_key_value
            new_key_value += 1

        sentences.append(sent)
        intents.append(intents_dct[label])

    if fit:
        X = tfidf.fit_transform(sentences)
    else:
        X = tfidf.transform(sentences)

    y = np.asarray(intents)

    return X, y


incomplete_path = '/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets'

for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
    print(f'Testing on: {dataset_size}')

    tfidf = TfidfVectorizer()
    intents_dct = {}
    new_key_value = 0

    # Intent classifier
    path_intents = os.path.join(incomplete_path, dataset_size, dataset_size + '.json')

    with open(path_intents) as f:
        int_ds = json.load(f)

    X_train, y_train = get_X_y(int_ds['train'] + int_ds['oos_test'], fit=True)  # fit only on first dataset
    X_test, y_test = get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

    mlp_int = MLPClassifier().fit(X_train, y_train)
    # ------------------------------------------

    # Results
    accuracy_correct = 0
    accuracy_out_of = 0

    recall_correct = 0
    recall_out_of = 0

    for sent_vec, true_int_label in zip(X_test, y_test):
        pred_int = mlp_int.predict(sent_vec)[0]  # intent prediction

        if true_int_label != intents_dct['oos']:
            if pred_int == true_int_label:
                accuracy_correct += 1

            accuracy_out_of += 1
        else:
            if pred_int == true_int_label:  # here pred_int is always oos
                recall_correct += 1

            recall_out_of += 1

    accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
    recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
    print(f'dataset_size: {dataset_size} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
