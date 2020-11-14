import fasttext, os
from testing import Testing
from utils import DS_INCOMPLETE_PATH

path_int = os.path.join(DS_INCOMPLETE_PATH, 'data_full', 'fasttext_labels', 'labels.')
DIM = 100  # dimension of pretrained vectors - either 100 or 300

# Train model for in-scope queries
model_int = fasttext.train_supervised(
    input=path_int + 'train', dim=DIM,
    pretrainedVectors=f'/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/pretrained_vectors/cc.en.{DIM}.vec')

for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
    print(f'Testing on: {dataset_size}')

    path_bin = os.path.join(DS_INCOMPLETE_PATH, dataset_size, 'fasttext_labels', 'labels.')

    X_test = []  # used to check correctness of results
    y_test = []

    with open(path_int + 'test_oos_test', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        X_test.append(message)
        y_test.append(label)

    # Train model for binary classification
    model_bin = fasttext.train_supervised(
        input=path_bin + 'train', dim=DIM,
        pretrainedVectors=f'/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/pretrained_vectors/cc.en.{DIM}.vec')

    # Test
    testing = Testing(model_int, X_test, y_test, 'fasttext', '__label__oos', bin_model=model_bin)
    results_dct = testing.test_binary()
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
