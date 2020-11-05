import fasttext, os
from testing import Testing

incomplete_path = '/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets'
path_int = os.path.join(incomplete_path, 'data_full', 'fasttext_labels', 'labels.')
DIM = 100  # dimension of pretrained vectors - either 100 or 300

# Train model for in-scope queries
model_int = fasttext.train_supervised(
    input=path_int + 'train', dim=DIM,
    pretrainedVectors=f'/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/pretrained_vectors/cc.en.{DIM}.vec')

for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
    print(f'Testing on: {dataset_size}')

    path_bin = os.path.join(incomplete_path, dataset_size, 'fasttext_labels', 'labels.')

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
    accuracy, recall = testing.test_binary()
    print(f'dataset_size: {dataset_size} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
