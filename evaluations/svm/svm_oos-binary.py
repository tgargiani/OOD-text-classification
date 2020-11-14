from sklearn import svm
import json, os
from testing import Testing
from utils import Split, DS_INCOMPLETE_PATH

# Intent classifier
path_intents = os.path.join(DS_INCOMPLETE_PATH, 'data_full', 'data_full.json')  # always use data_full dataset

with open(path_intents) as f:
    int_ds = json.load(f)

split = Split()

X_int_train, y_int_train = split.get_X_y(int_ds['train'], fit=True)  # fit only on first dataset
X_int_test, y_int_test = split.get_X_y(int_ds['test'] + int_ds['oos_test'], fit=False)

svc_int = svm.SVC(C=1, kernel='linear').fit(X_int_train, y_int_train)
# ------------------------------------------

for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
    print(f'Testing on: {dataset_size}')

    # Binary classifier
    path_bin = os.path.join(DS_INCOMPLETE_PATH, dataset_size, dataset_size + '.json')

    with open(path_bin) as f:
        bin_ds = json.load(f)

    X_bin_train, y_bin_train = split.get_X_y(bin_ds['train'], fit=False)
    # X_bin_test, y_bin_test = split.get_X_y(bin_ds['test'], fit=False)

    svc_bin = svm.SVC(C=1, kernel='linear').fit(X_bin_train, y_bin_train)
    # ------------------------------------------

    # Test
    testing = Testing(svc_int, X_int_test, y_int_test, 'svm', split.intents_dct['oos'], bin_model=svc_bin)
    results_dct = testing.test_binary()
    print(
        f'dataset_size: {dataset_size} -- '
        f'accuracy: {round(results_dct["accuracy"], 1)}, '
        f'recall: {round(results_dct["recall"], 1)}, '
        f'far: {round(results_dct["far"], 1)}, '
        f'frr: {round(results_dct["frr"], 1)}\n')
