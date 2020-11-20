import fasttext, os, json, copy
from testing import Testing
from utils import get_intents_selection, get_filtered_lst, print_results, dataset_2_string, get_X_y_fasttext, \
    DS_INCOMPLETE_PATH, PRETRAINED_VECTORS_PATH
from tempfile import NamedTemporaryFile
from numpy import mean


def evaluate(binary_dataset, model_int, X_int_test, y_int_test, dim):
    train_str_bin = dataset_2_string(binary_dataset['train'])
    # X_bin_test, y_bin_test = get_X_y_fasttext(binary_dataset['test'])

    with NamedTemporaryFile() as f:
        f.write(train_str_bin.encode('utf8'))
        f.seek(0)

        # Train model for in-scope queries
        model_bin = fasttext.train_supervised(
            input=f.name, dim=dim,
            pretrainedVectors=f'{PRETRAINED_VECTORS_PATH}/cc.en.{dim}.vec')

    # Test
    testing = Testing(model_int, X_int_test, y_int_test, 'fasttext', '__label__oos', bin_model=model_bin)
    results_dct = testing.test_binary()

    return results_dct


if __name__ == '__main__':
    DIM = 100  # dimension of pre-trained vectors - either 100 or 300
    RANDOM_SELECTION = False  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection

    # Intent classifier
    path_intents = os.path.join(DS_INCOMPLETE_PATH, 'data_full', 'data_full.json')  # always use data_full dataset

    with open(path_intents) as f:
        int_ds = json.load(f)

    train_str_int = dataset_2_string(int_ds['train'])
    X_int_test, y_int_test = get_X_y_fasttext(int_ds['test'] + int_ds['oos_test'])

    with NamedTemporaryFile() as f:
        f.write(train_str_int.encode('utf8'))
        f.seek(0)

        # Train model for in-scope queries
        model_int = fasttext.train_supervised(
            input=f.name, dim=DIM,
            pretrainedVectors=f'{PRETRAINED_VECTORS_PATH}/cc.en.{DIM}.vec')

    for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
        print(f'Testing on: {dataset_size}\n')

        # Binary classifier
        path_bin = os.path.join(DS_INCOMPLETE_PATH, dataset_size, dataset_size + '.json')

        with open(path_bin) as f:  # open binary intent dataset
            bin_ds = json.load(f)

        if not RANDOM_SELECTION:
            results_dct = evaluate(bin_ds, model_int, X_int_test, y_int_test, DIM)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    selection = get_intents_selection(bin_ds['train'],
                                                      num_samples)  # selected intent labels: (num_samples, ) np.ndarray

                    filt_train = get_filtered_lst(bin_ds['train'],
                                                  selection)  # almost the same as int_ds['train'] but filtered according to selection
                    # filt_test = get_filtered_lst(bin_ds['test'], selection)

                    mod_bin_ds = copy.deepcopy(bin_ds)  # deepcopy in order to not modify the original dict
                    mod_bin_ds['train'] = filt_train
                    # mod_bin_ds['test'] = filt_test

                    temp_res = evaluate(mod_bin_ds, model_int, X_int_test, y_int_test, DIM)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
