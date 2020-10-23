import json, os

def create_fasttext_datasets(path):
    path_head, path_tail = os.path.split(path)

    with open(path) as f:
        intents = json.load(f)

    train, val, test, oos_train, oos_val, oos_test = None, None, None, None, None, None
    dct = {'train': train, 'val': val, 'test': test, 'oos_train': oos_train, 'oos_val': oos_val, 'oos_test': oos_test}

    for var_name, var in dct.items():
        var = open(os.path.join(path_head, 'fasttext_labels', 'labels.' + var_name), 'w+')

        for query, intent in intents[var_name]:
            line = '__label__' + intent + ' ' + query + '\n'
            var.write(line)

        var.close()

        if var_name in ['train', 'val', 'test']:
            varname_oos_varname = var_name + '_oos_' + var_name
            var = open(os.path.join(path_head, 'fasttext_labels', 'labels.' + varname_oos_varname), 'w+')

            for query, intent in intents[var_name]:
                line = '__label__' + intent + ' ' + query + '\n'
                var.write(line)

            for query, intent in intents['oos_' + var_name]:
                line = '__label__' + intent + ' ' + query + '\n'
                var.write(line)

            var.close()

# create_fasttext_datasets('/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets/data_full/data_full.json')
# create_fasttext_datasets('/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets/data_imbalanced/data_imbalanced.json')
# create_fasttext_datasets('/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets/data_oos_plus/data_oos_plus.json')
# create_fasttext_datasets('/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets/data_small/data_small.json')
