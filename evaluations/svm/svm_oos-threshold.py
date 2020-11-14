from sklearn import svm
import json, os
import numpy as np
from testing import Testing
from utils import Split, DS_INCOMPLETE_PATH

for dataset_size in ['data_full', 'data_small', 'data_imbalanced']:
    print(f'Testing on: {dataset_size}')

    split = Split()

    # Intent classifier
    path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size, dataset_size + '.json')

    with open(path_intents) as f:
        int_ds = json.load(f)

    X_train, y_train = split.get_X_y(int_ds['train'], fit=True)  # fit only on first dataset
    X_val, y_val = split.get_X_y(int_ds['val'] + int_ds['oos_val'], fit=False)
    X_test, y_test = split.get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

    svc_int = svm.SVC(C=1, kernel='linear', probability=True).fit(X_train, y_train)

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
                pred_label = split.intents_dct['oos']

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

    # Test
    testing = Testing(svc_int, X_test, y_test, 'svm', split.intents_dct['oos'])
    results_dct = testing.test_threshold(threshold)
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
