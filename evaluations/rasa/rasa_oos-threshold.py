import rasa, os, json, copy
from testing import Testing
from utils import get_intents_selection, get_filtered_lst, print_results, dataset_2_string_rasa, get_X_y_rasa, \
    find_best_threshold, save_results, DS_INCOMPLETE_PATH
from tempfile import NamedTemporaryFile
from numpy import mean


def evaluate(dataset, limit_num_sents: bool):
    train_str = dataset_2_string_rasa(dataset['train'], limit_num_sents=limit_num_sents, set_type='train')
    X_val, y_val = get_X_y_rasa(dataset['val'] + dataset['oos_val'], limit_num_sents=limit_num_sents,
                                set_type='val')
    X_test, y_test = get_X_y_rasa(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                  set_type='test')

    with NamedTemporaryFile(suffix='.yml') as f:
        f.write(train_str.encode('utf8'))
        f.seek(0)

        training_data = rasa.shared.nlu.training_data.loading.load_data(f.name)

    config = rasa.nlu.config.load('config.yml')
    trainer = rasa.nlu.model.Trainer(config)
    model = trainer.train(training_data)

    val_predictions_labels = []  # used to find threshold

    for sent, true_int_label in zip(X_val, y_val):
        pred = model.parse(sent)
        pred_label = pred['intent']['name']
        similarity = pred['intent']['confidence']

        pred = (pred_label, similarity)
        val_predictions_labels.append((pred, true_int_label))

    threshold = find_best_threshold(val_predictions_labels, 'oos')

    # Test
    testing = Testing(model, X_test, y_test, 'rasa', 'oos')
    results_dct = testing.test_threshold(threshold)

    return results_dct


if __name__ == '__main__':
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    LIMIT_NUM_SENTS = True  # am I limiting the number of sentences of each intent?

    for dataset_size in ['data_full', 'data_small', 'data_imbalanced']:
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
                    filt_val = get_filtered_lst(int_ds['val'], selection)
                    filt_test = get_filtered_lst(int_ds['test'], selection)

                    mod_int_ds = copy.deepcopy(int_ds)  # deepcopy in order to not modify the original dict
                    mod_int_ds['train'] = filt_train
                    mod_int_ds['val'] = filt_val
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

                # save_results('rasa', 'oos-threshold', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
