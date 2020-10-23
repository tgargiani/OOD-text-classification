from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import numpy as np

tfidf = TfidfVectorizer()
intents_dct = {}
new_key_value = 0


def get_X_y(lst, fit=True):
    """Splits the 2D list into sentences and intents."""

    global tfidf
    global intents_dct
    global new_key_value

    sentences = []
    intents = []

    for sent, inte in lst:
        if inte not in intents_dct.keys():
            intents_dct[inte] = new_key_value
            new_key_value += 1

        sentences.append(sent)
        intents.append(intents_dct[inte])

    if fit:
        X = tfidf.fit_transform(sentences)
    else:
        X = tfidf.transform(sentences)

    y = np.asarray(intents)

    return X, y


# Intent classifier
path_intents = '/Users/tommaso.gargiani/Documents/FEL/PROJ/datasets/data_full/data_full.json'

with open(path_intents) as f:
    int_ds = json.load(f)

X_int_train, y_int_train = get_X_y(int_ds['train'], fit=True)
X_int_test, y_int_test = get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

svc_int = svm.SVC().fit(X_int_train, y_int_train)
# ------------------------------------------

# Binary classifier
path_bin = '/Users/tommaso.gargiani/Documents/FEL/PROJ/datasets/binary_undersample/binary_undersample.json'

with open(path_bin) as f:
    bin_ds = json.load(f)

X_bin_train, y_bin_train = get_X_y(bin_ds['train'], fit=False)
X_bin_test, y_bin_test = get_X_y(bin_ds['test'], fit=False)

svc_bin = svm.SVC().fit(X_bin_train, y_bin_train)
# ------------------------------------------

# Results
accuracy_correct = 0
accuracy_out_of = 0

recall_correct = 0
recall_out_of = 0

for sent_vec, true_int_label in zip(X_int_test, y_int_test):
    pred_bin = svc_bin.predict(sent_vec)[0]  # binary prediction

    if pred_bin == intents_dct['in']:
        pred_int = svc_int.predict(sent_vec)[0]  # intent prediction

        if pred_int == true_int_label:
            accuracy_correct += 1

        accuracy_out_of += 1
    else:
        if pred_bin == true_int_label:  # here pred_bin is always oos
            recall_correct += 1

        recall_out_of += 1

accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
print(f'dataset_size: {"binary_undersample"} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
