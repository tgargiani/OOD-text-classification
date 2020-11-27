from sklearn import svm
import json, os, copy
from testing import Testing
from utils import Split, get_intents_selection, get_filtered_lst, print_results, save_results, DS_INCOMPLETE_PATH
from numpy import mean


def evaluate(binary_dataset, svc_int, X_int_test, y_int_test, split):
    X_bin_train, y_bin_train = split.get_X_y(binary_dataset['train'], fit=False)
    # X_bin_test, y_bin_test = split.get_X_y(binary_dataset['test'], fit=False)

    svc_bin = svm.SVC(C=1, kernel='linear').fit(X_bin_train, y_bin_train)

    # Test
    testing = Testing(svc_int, X_int_test, y_int_test, 'svm', split.intents_dct['oos'], bin_model=svc_bin)
    results_dct = testing.test_binary()

    return results_dct


def train_intent_model(int_ds, random_selection: bool, num_samples=None):
    if random_selection:
        selection = get_intents_selection(int_ds['train'],
                                          num_samples)  # selected intent labels: (num_samples, ) np.ndarray

        filt_train = get_filtered_lst(int_ds['train'],
                                      selection)  # almost the same as int_ds['train'] but filtered according to selection
        filt_test = get_filtered_lst(int_ds['test'], selection)

        mod_int_ds = copy.deepcopy(int_ds)  # deepcopy in order to not modify the original dict
        mod_int_ds['train'] = filt_train
        mod_int_ds['test'] = filt_test

        dataset = mod_int_ds
    else:
        dataset = int_ds

    split = Split()

    X_int_train, y_int_train = split.get_X_y(dataset['train'], fit=True)  # fit only on first dataset
    X_int_test, y_int_test = split.get_X_y(dataset['test'] + dataset['oos_test'], fit=False)

    svc_int = svm.SVC(C=1, kernel='linear').fit(X_int_train, y_int_train)

    return svc_int, X_int_test, y_int_test, split


if __name__ == '__main__':
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection

    # Intent classifier
    path_intents = os.path.join(DS_INCOMPLETE_PATH, 'data_full.json')  # always use data_full dataset

    with open(path_intents) as f:
        int_ds = json.load(f)

    for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
        print(f'Testing on: {dataset_size}\n')

        # Binary classifier
        path_bin = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_bin) as f:  # open binary intent dataset
            bin_ds = json.load(f)

        if not RANDOM_SELECTION:
            svc_int, X_int_test, y_int_test, split = train_intent_model(int_ds, RANDOM_SELECTION)

            results_dct = evaluate(bin_ds, svc_int, X_int_test, y_int_test, split)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    svc_int, X_int_test, y_int_test, split = train_intent_model(int_ds, RANDOM_SELECTION, num_samples)

                    temp_res = evaluate(bin_ds, svc_int, X_int_test, y_int_test, split)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('svm', 'oos-binary', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
