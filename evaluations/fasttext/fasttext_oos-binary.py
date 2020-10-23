import fasttext, os

incomplete_path = '/Users/tommaso.gargiani/Documents/FEL/PROJ/datasets'
path_int = os.path.join(incomplete_path, 'data_full', 'fasttext_labels', 'labels.')

# train model for in-scope queries
# model_int = fasttext.train_supervised(
#     input=path_int + 'train', dim=100,
#     pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.100.vec')

model_int = fasttext.train_supervised(
    input=path_int + 'train', dim=300,
    pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.300.vec')

for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
    print(f'Testing on: {dataset_size}')

    path_bin = os.path.join(incomplete_path, dataset_size, 'fasttext_labels', 'labels.')

    test_true = []  # used to check correctness of results

    with open(path_int + 'test_oos_test', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        test_true.append((label, message))

    # train model for binary classification
    # model_bin = fasttext.train_supervised(
    #     input=path_bin + 'train', dim=100,
    #     pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.100.vec')

    model_bin = fasttext.train_supervised(
        input=path_bin + 'train', dim=300,
        pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.300.vec')

    accuracy_correct = 0
    accuracy_out_of = 0

    recall_correct = 0
    recall_out_of = 0

    for label, message in test_true:
        # 1st step - binary classification
        bin_pred = model_bin.predict(message)
        bin_pred_label = bin_pred[0][0]
        bin_similarity = bin_pred[1][0]

        if bin_pred_label == '__label__in':
            # 2nd step - intent classification
            int_pred = model_int.predict(message)
            int_pred_label = int_pred[0][0]
            int_similarity = int_pred[1][0]

            if int_pred_label == label:
                accuracy_correct += 1

            accuracy_out_of += 1
        else:
            if bin_pred_label == label:  # here bin_pred_label is always __label__oos
                recall_correct += 1

            recall_out_of += 1

    accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
    recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
    print(f'dataset_size: {dataset_size} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
