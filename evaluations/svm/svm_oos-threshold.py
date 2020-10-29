from sklearn import svm
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

for dataset_size in ['data_full', 'data_small', 'data_imbalanced']:
    print(f'Testing on: {dataset_size}')

    tfidf = TfidfVectorizer()
    intents_dct = {}
    new_key_value = 0

    # Intent classifier
    path_intents = os.path.join(incomplete_path, dataset_size, dataset_size + '.json')

    with open(path_intents) as f:
        int_ds = json.load(f)

    X_train, y_train = get_X_y(int_ds['train'], fit=True)  # fit only on first dataset
    X_val, y_val = get_X_y(int_ds['val'] + int_ds['oos_val'], fit=False)
    X_test, y_test = get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

    svc_int = svm.SVC(probability=True).fit(X_train, y_train)

    val_predictions_labels = []  # used to find threshold

    for sent_vec, true_int_label in zip(X_val, y_val):
        pred_probs = svc_int.predict_proba(sent_vec)[0]  # intent prediction probabilities
        pred_label = np.argmax(pred_probs)  # intent prediction
        similarity = pred_probs[pred_label]

        pred = (pred_label, similarity)
        val_predictions_labels.append((pred, true_int_label))

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
                pred_label = intents_dct['oos']

            if pred_label == label:
                val_accuracy_correct += 1

            val_accuracy_out_of += 1

        val_accuracy = (val_accuracy_correct / val_accuracy_out_of) * 100

        if val_accuracy < previous_val_accuracy:
            threshold = thresholds[idx - 1]  # best threshold is the previous one
            break

        previous_val_accuracy = val_accuracy
        threshold = tr

    # ------------------------------------------

    # Results
    accuracy_correct = 0
    accuracy_out_of = 0

    recall_correct = 0
    recall_out_of = 0

    for sent_vec, true_int_label in zip(X_test, y_test):
        pred_probs = svc_int.predict_proba(sent_vec)[0]  # intent prediction probabilities
        pred_label = np.argmax(pred_probs)  # intent prediction
        similarity = pred_probs[pred_label]

        if similarity < threshold:
            pred_label = intents_dct['oos']

        if true_int_label != intents_dct['oos']:
            if pred_label == true_int_label:
                accuracy_correct += 1

            accuracy_out_of += 1
        else:
            if pred_label == true_int_label:  # here pred_label is always oos
                recall_correct += 1

            recall_out_of += 1

    accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
    recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
    print(f'dataset_size: {dataset_size} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
