import fasttext, os, json, copy
from testing import Testing
from utils import get_intents_selection, get_filtered_lst, print_results, dataset_2_string, get_X_y_fasttext, \
    save_results, DS_INCOMPLETE_PATH, PRETRAINED_VECTORS_PATH
from tempfile import NamedTemporaryFile
from numpy import mean


def evaluate(dataset, dim: int, limit_num_sents: bool):
    train_str = dataset_2_string(dataset['train'] + dataset['oos_train'], limit_num_sents=limit_num_sents,
                                 set_type='train')
    X_test, y_test = get_X_y_fasttext(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                      set_type='test')

    with NamedTemporaryFile() as f:
        f.write(train_str.encode('utf8'))
        f.seek(0)

        # Train model for in-scope queries
        model = fasttext.train_supervised(
            input=f.name, dim=dim,
            pretrainedVectors=f'{PRETRAINED_VECTORS_PATH}/cc.en.{dim}.vec')

    # Test
    testing = Testing(model, X_test, y_test, 'fasttext', '__label__oos')
    results_dct = testing.test_train()

    return results_dct


if __name__ == '__main__':
    DIM = 100  # dimension of pre-trained vectors - either 100 or 300
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    LIMIT_NUM_SENTS = True  # am I limiting the number of sentences of each intent?

    print(f'DIM: {DIM}')

    for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
        print(f'Testing on: {dataset_size}')

        path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_intents) as f:  # open intent dataset
            int_ds = json.load(f)

        if not RANDOM_SELECTION:
            results_dct = evaluate(int_ds, DIM, LIMIT_NUM_SENTS)

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

                    temp_res = evaluate(mod_int_ds, DIM, LIMIT_NUM_SENTS)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('fasttext', 'oos-train', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
