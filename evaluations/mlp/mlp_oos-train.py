from sklearn.neural_network import MLPClassifier
import json, os
from testing import Testing
from utils import Split, DS_INCOMPLETE_PATH

for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
    print(f'Testing on: {dataset_size}')

    split = Split()

    # Intent classifier
    path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size, dataset_size + '.json')

    with open(path_intents) as f:
        int_ds = json.load(f)

    X_train, y_train = split.get_X_y(int_ds['train'] + int_ds['oos_train'], fit=True)  # fit only on first dataset
    X_test, y_test = split.get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

    mlp_int = MLPClassifier(activation='tanh').fit(X_train, y_train)
    # ------------------------------------------

    # Test
    testing = Testing(mlp_int, X_test, y_test, 'mlp', split.intents_dct['oos'])
    results_dct = testing.test_train()
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
