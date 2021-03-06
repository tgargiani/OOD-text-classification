from sklearn import svm
import json, os, copy
from testing import Testing
from utils import Split, get_intents_selection, get_filtered_lst, print_results, save_results, DS_INCOMPLETE_PATH
from numpy import mean


def evaluate(dataset, limit_num_sents: bool):
    # Split dataset
    split = Split()

    X_train, y_train = split.get_X_y(dataset['train'] + dataset['oos_train'], fit=True,
                                     limit_num_sents=limit_num_sents, set_type='train')  # fit only on first dataset
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'], fit=False, limit_num_sents=limit_num_sents,
                                   set_type='test')

    svc_int = svm.SVC(C=1, kernel='linear').fit(X_train, y_train)

    # Test
    testing = Testing(svc_int, X_test, y_test, 'svm', split.intents_dct['oos'])
    results_dct = testing.test_train()

    return results_dct


if __name__ == '__main__':
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    LIMIT_NUM_SENTS = True  # am I limiting the number of sentences of each intent?

    for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
        print(f'Testing on: {dataset_size}\n')

        path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_intents) as f:  # open intent dataset
            int_ds = json.load(f)

        if not RANDOM_SELECTION:
            results_dct = evaluate(int_ds, LIMIT_NUM_SENTS)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    selection = get_intents_selection(int_ds['train'],
                                                      num_samples)  # selected intent labels: (num_samples, ) np.ndarray

                    filt_train = get_filtered_lst(int_ds['train'],
                                                  selection)  # almost the same as int_ds['train'] but filtered according to selection
                    filt_test = get_filtered_lst(int_ds['test'], selection)

                    mod_int_ds = copy.deepcopy(int_ds)  # deepcopy in order to not modify the original dict
                    mod_int_ds['train'] = filt_train
                    mod_int_ds['test'] = filt_test

                    temp_res = evaluate(mod_int_ds, LIMIT_NUM_SENTS)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('svm', 'oos-train', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
