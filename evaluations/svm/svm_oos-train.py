from sklearn import svm
import json, os
from testing import Testing
from utils import Utils

incomplete_path = '/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets'

for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
    print(f'Testing on: {dataset_size}')

    utils = Utils()

    # Intent classifier
    path_intents = os.path.join(incomplete_path, dataset_size, dataset_size + '.json')

    with open(path_intents) as f:
        int_ds = json.load(f)

    X_train, y_train = utils.get_X_y(int_ds['train'] + int_ds['oos_test'], fit=True)  # fit only on first dataset
    X_test, y_test = utils.get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

    svc_int = svm.SVC().fit(X_train, y_train)
    # ------------------------------------------

    # Test
    testing = Testing(svc_int, X_test, y_test, 'svm', utils.intents_dct['oos'])
    results_dct = testing.test_train()
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
