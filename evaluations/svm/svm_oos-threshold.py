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


# Intent classifier
path_intents = '/Users/tommaso.gargiani/Documents/FEL/PROJ/datasets/data_full/data_full.json'

with open(path_intents) as f:
    int_ds = json.load(f)

X_train, y_train = get_X_y(int_ds['train'], fit=True)
X_val, y_val = get_X_y(int_ds['val'] + int_ds['oos_val'], fit=False)
X_test, y_test = get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

svc_int = svm.SVC(probability=True).fit(X_train, y_train)

val_predictions_labels = []

for sent_vec, true_int_label in zip(X_val, y_val):
    pred_int = svc_int.predict(sent_vec)[0]  # intent prediction
    pred_proba = svc_int.predict_proba(sent_vec)[0]  # intent prediction
    print(pred_int, pred_proba, np.argmax(pred_proba), np.min(pred_proba))
    break
# # ------------------------------------------
#
# # Results
# accuracy_correct = 0
# accuracy_out_of = 0
#
# recall_correct = 0
# recall_out_of = 0
#
# for sent_vec, true_int_label in zip(X_test, y_test):
#     pred_int = svc_int.predict(sent_vec)[0]  # intent prediction
#
#     if true_int_label != intents_dct['oos']:
#         if pred_int == true_int_label:
#             accuracy_correct += 1
#
#         accuracy_out_of += 1
#     else:
#         if pred_int == true_int_label:  # here pred_int is always oos
#             recall_correct += 1
#
#         recall_out_of += 1
#
# accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
# recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
# print(f'dataset_size: {"binary_undersample"} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
