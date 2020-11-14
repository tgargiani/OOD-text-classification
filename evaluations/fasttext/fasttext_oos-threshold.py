import fasttext, os
import numpy as np
from testing import Testing
from utils import DS_INCOMPLETE_PATH

DIM = 100  # dimension of pretrained vectors - either 100 or 300

for dataset_size in ['data_full', 'data_small', 'data_imbalanced']:
    print(f'Testing on: {dataset_size}')

    path = os.path.join(DS_INCOMPLETE_PATH, dataset_size, 'fasttext_labels', 'labels.')
    val_true = []  # used to find the validation threshold

    with open(path + 'val_oos_val', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        val_true.append((label, message))

    X_test = []  # used to check correctness of results
    y_test = []

    with open(path + 'test_oos_test', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        X_test.append(message)
        y_test.append(label)

    # Train model for in-scope queries
    model = fasttext.train_supervised(
        input=path + 'train', dim=DIM,
        pretrainedVectors=f'/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/pretrained_vectors/cc.en.{DIM}.vec')

    val_predictions_labels = []  # used to find threshold

    for label, message in val_true:
        pred = model.predict(message)
        val_predictions_labels.append((pred, label))

    # Initialize search for best threshold
    thresholds = np.linspace(0, 1, 101)  # all possible thresholds
    previous_val_accuracy = 0
    threshold = 0

    # Find best threshold
    for idx, tr in enumerate(thresholds):
        val_accuracy_correct = 0
        val_accuracy_out_of = 0

        for pred, label in val_predictions_labels:
            pred_label = pred[0][0]
            similarity = pred[1][0]

            if similarity < tr:
                pred_label = '__label__oos'

            if pred_label == label:
                val_accuracy_correct += 1

            val_accuracy_out_of += 1

        val_accuracy = (val_accuracy_correct / val_accuracy_out_of) * 100

        if val_accuracy < previous_val_accuracy:
            threshold = thresholds[idx - 1]  # best threshold is the previous one
            break

        previous_val_accuracy = val_accuracy
        threshold = tr

    # Test
    testing = Testing(model, X_test, y_test, 'fasttext', '__label__oos')
    results_dct = testing.test_threshold(threshold)
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
