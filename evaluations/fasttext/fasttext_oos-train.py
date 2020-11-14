import fasttext, os
from testing import Testing
from utils import DS_INCOMPLETE_PATH

DIM = 100  # dimension of pretrained vectors - either 100 or 300

for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
    print(f'Testing on: {dataset_size}')

    path = os.path.join(DS_INCOMPLETE_PATH, dataset_size, 'fasttext_labels', 'labels.')

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
        input=path + 'train_oos_train', dim=DIM,
        pretrainedVectors=f'/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/pretrained_vectors/cc.en.{DIM}.vec')

    # Test
    testing = Testing(model, X_test, y_test, 'fasttext', '__label__oos')
    results_dct = testing.test_train()
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
