import json, os

def create_in_oos_fasttext_datasets(path):
    path_head, path_tail = os.path.split(path)

    with open(path) as f:
        intents = json.load(f)

    train, val, test = None, None, None
    dct = {'train': train, 'val': val, 'test': test}

    for var_name, var in dct.items():
        var = open(os.path.join(path_head, 'fasttext_labels', 'labels.' + var_name), 'w+')

        for query, intent in intents[var_name]:
            line = '__label__' + intent + ' ' + query + '\n'
            var.write(line)

        var.close()

# create_in_oos_fasttext_datasets(
#     '/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets/binary_undersample/binary_undersample.json')
# create_in_oos_fasttext_datasets(
#     '/Users/tommaso.gargiani/Documents/FEL/OOD-text-classification/datasets/binary_wiki_aug/binary_wiki_aug.json')
