import fasttext, os

incomplete_path = '/Users/tommaso.gargiani/Documents/FEL/PROJ/datasets'

for dataset_size in ['data_full', 'data_small', 'data_imbalanced', 'data_oos_plus']:
    print(f'Testing on: {dataset_size}')

    path = os.path.join(incomplete_path, dataset_size, 'fasttext_labels', 'labels.')
    test_true = []  # used to check correctness of results

    with open(path + 'test_oos_test', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        test_true.append((label, message))

    # model = fasttext.train_supervised(
    #     input=path + 'train_oos_train', dim=100,
    #     pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.100.vec')

    model = fasttext.train_supervised(
        input=path + 'train_oos_train', dim=300,
        pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.300.vec')

    accuracy_correct = 0
    accuracy_out_of = 0

    recall_correct = 0
    recall_out_of = 0

    for label, message in test_true:
        pred = model.predict(message)
        pred_label = pred[0][0]
        similarity = pred[1][0]

        # print(best_label, similarity, pred, message)

        if label != '__label__oos':  # measure accuracy
            if pred_label == label:
                accuracy_correct += 1

            accuracy_out_of += 1
        else:  # measure recall
            if pred_label == label:
                recall_correct += 1

            recall_out_of += 1

    accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
    recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
    print(f'dataset_size: {dataset_size} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
