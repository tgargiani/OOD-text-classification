import fasttext, os, json, copy
from testing import Testing
from utils import get_intents_selection, get_filtered_lst, print_results, dataset_2_string, get_X_y_fasttext, \
    save_results, DS_INCOMPLETE_PATH, PRETRAINED_VECTORS_PATH
from tempfile import NamedTemporaryFile
from numpy import mean


def evaluate(binary_dataset, model_int, X_int_test, y_int_test, dim):
    train_str_bin = dataset_2_string(binary_dataset['train'])

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


def train_intent_model(int_ds, random_selection: bool, dim: int, num_samples=None):
    if random_selection:
        selection = get_intents_selection(int_ds['train'],
                                          num_samples)  # selected intent labels: (num_samples, ) np.ndarray

        filt_train = get_filtered_lst(int_ds['train'],
                                      selection)  # almost the same as int_ds['train'] but filtered according to selection

        mod_int_ds = copy.deepcopy(int_ds)  # deepcopy in order to not modify the original dict
        mod_int_ds['train'] = filt_train

        dataset = mod_int_ds
    else:
        dataset = int_ds

    train_str_int = dataset_2_string(dataset['train'])
    X_int_test, y_int_test = get_X_y_fasttext(dataset['test'] + dataset['oos_test'])

    with NamedTemporaryFile() as f:
        f.write(train_str_int.encode('utf8'))
        f.seek(0)

        # Train model for in-scope queries
        model_int = fasttext.train_supervised(
            input=f.name, dim=dim,
            pretrainedVectors=f'{PRETRAINED_VECTORS_PATH}/cc.en.{dim}.vec')

    return model_int, X_int_test, y_int_test


if __name__ == '__main__':
    DIM = 100  # dimension of pre-trained vectors - either 100 or 300
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    print(f'DIM: {DIM}')

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
            model_int, X_int_test, y_int_test = train_intent_model(int_ds, RANDOM_SELECTION, DIM)

            results_dct = evaluate(bin_ds, model_int, X_int_test, y_int_test, DIM)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    model_int, X_int_test, y_int_test = train_intent_model(int_ds, RANDOM_SELECTION, DIM, num_samples)

                    temp_res = evaluate(bin_ds, model_int, X_int_test, y_int_test, DIM)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('fasttext', 'oos-binary', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
