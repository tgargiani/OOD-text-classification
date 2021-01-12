import rasa, os, json, copy
from testing import Testing
from utils import get_intents_selection, get_filtered_lst, print_results, dataset_2_string_rasa, get_X_y_rasa, \
    save_results, DS_INCOMPLETE_PATH
from tempfile import NamedTemporaryFile
from numpy import mean


def evaluate(binary_dataset, model_int, X_int_test, y_int_test):
    train_str_bin = dataset_2_string_rasa(binary_dataset['train'], limit_num_sents=False, set_type='train')

    with NamedTemporaryFile(suffix='.yml') as f:
        f.write(train_str_bin.encode('utf8'))
        f.seek(0)

        training_data = rasa.shared.nlu.training_data.loading.load_data(f.name)

    config = rasa.nlu.config.load('config.yml')
    trainer = rasa.nlu.model.Trainer(config)
    model_bin = trainer.train(training_data)

    # Test
    testing = Testing(model_int, X_int_test, y_int_test, 'rasa', 'oos', bin_model=model_bin)
    results_dct = testing.test_binary()

    return results_dct


def train_intent_model(int_ds, random_selection: bool, limit_num_sents: bool, num_samples=None):
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

    train_str_int = dataset_2_string_rasa(dataset['train'], limit_num_sents=limit_num_sents, set_type='train')
    X_int_test, y_int_test = get_X_y_rasa(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                          set_type='test')

    with NamedTemporaryFile(suffix='.yml') as f:
        f.write(train_str_int.encode('utf8'))
        f.seek(0)

        training_data = rasa.shared.nlu.training_data.loading.load_data(f.name)

    config = rasa.nlu.config.load('config.yml')
    trainer = rasa.nlu.model.Trainer(config)
    model_int = trainer.train(training_data)

    return model_int, X_int_test, y_int_test


if __name__ == '__main__':
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    LIMIT_NUM_SENTS = True  # am I limiting the number of sentences of each intent?

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
            model_int, X_int_test, y_int_test = train_intent_model(int_ds, RANDOM_SELECTION, LIMIT_NUM_SENTS)

            results_dct = evaluate(bin_ds, model_int, X_int_test, y_int_test)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    model_int, X_int_test, y_int_test = train_intent_model(int_ds, RANDOM_SELECTION, LIMIT_NUM_SENTS,
                                                                           num_samples)

                    temp_res = evaluate(bin_ds, model_int, X_int_test, y_int_test)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('rasa', 'oos-binary', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
