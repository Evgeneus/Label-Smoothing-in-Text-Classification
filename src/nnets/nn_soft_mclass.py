import torch
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from src.nnets.model import MLP1
from src.nnets.data_loader import load_data_mlp
from src.nnets.utils import CrossEntropyLossSoft
from src.nnets.utils import ece_score, plot_reliability_diagram


torch_seed = 2020
torch.manual_seed(torch_seed)


def train_neural_net(net_params, tolerance=20):
    # define NNet and training process
    lr_rate = net_params['lr_rate']
    weight_decay = net_params['weight_decay']
    class_weight = net_params['class_weight']

    model = MLP1(input_dim, output_dim)
    criterion = CrossEntropyLossSoft(class_weight)
    criterion_val = CrossEntropyLossSoft()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    # init early stopping
    t = 1
    err = float('inf')
    val_stat = []

    # check if binary or multi class classification
    num_classes = len(y_train_hard.unique())
    if num_classes == 2:
        average = 'binary'
    else:
        average = 'macro'

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tfidf)
        loss = criterion(outputs, y_train_soft)
        loss.backward()
        optimizer.step()

        # evaluate model on val data
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tfidf)
            loss_val = criterion_val(outputs_val, y_val_soft).item()
            ece_val = ece_score(y_val_hard.numpy(), torch.sigmoid(outputs_val).numpy())
            _, y_pred = torch.max(torch.sigmoid(outputs_val).data, 1)
            pre_val, rec_val, f1_val, _ = precision_recall_fscore_support(y_val_hard, y_pred, average=average, beta=1)
            acc_val = accuracy_score(y_val_hard, y_pred)
        val_stat.append([loss_val, pre_val, rec_val, acc_val, epoch, ece_val, f1_val])

        # early stopping
        if epoch > 40:
            if round(loss_val, 3) < err:
                t = 1
                err = loss_val
            else:
                if t == tolerance:
                    break
                t += 1
    print(
        'Epoch {}. Train Loss : {:1.3f} | Val Loss: {:1.3f} | Val ECE: {:1.3f} | Val F1: {:1.4f} | Val Acc: {:1.3f}'
        'Val P: {:1.3f} | Val R: {:1.3f}'.format(epoch, loss, loss_val, ece_val, f1_val, pre_val, rec_val, acc_val))

    return val_stat[epoch - tolerance]


def train_evaluate(net_params):
    # define NNet and training process
    lr_rate = net_params['lr_rate']
    weight_decay = net_params['weight_decay']
    class_weight = net_params['class_weight']
    epochs = net_params['epochs']

    model = MLP1(input_dim, output_dim)
    criterion = CrossEntropyLossSoft(class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    # train model
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tfidf)
        loss = criterion(outputs, y_train_soft)
        loss.backward()
        optimizer.step()
        print("Epoch {}. Train Loss : {:1.3f}".format(epoch, loss))

    # evaluate model on test data
    with torch.no_grad():
        outputs_test = model(X_test_tfidf)
        CrossEntropyLossHard = torch.nn.CrossEntropyLoss()
        loss_test_hard = CrossEntropyLossHard(outputs_test, y_test_hard).item()
        # Get predictions from the maximum value
        _, y_pred = torch.max(torch.sigmoid(outputs_test).data, 1)
    ece_test = ece_score(y_test_hard.numpy(), torch.sigmoid(outputs_test).numpy())
    # check if binary or multi class classification
    num_classes = len(y_test_hard.unique())
    if num_classes == 2:
        average = 'binary'
    else:
        average = 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_hard, y_pred, average=average, beta=1)
    _, _, f01, _ = precision_recall_fscore_support(y_test_hard, y_pred, average=average, beta=0.1)
    _, _, f10, _ = precision_recall_fscore_support(y_test_hard, y_pred, average=average, beta=10)

    plot_reliability_diagram(y_test_hard.numpy(), torch.sigmoid(outputs_test).numpy(), title_suffix='NN-SemiHard (ECE={:1.4f})'.format(ece_test))
    print('------------')
    print('*Evaluation on test data (SemiSoft), epoch {}*'.format(epoch))
    print('Test ECE: {:1.4f}'.format(ece_test))
    print('Test Loss Hard: {:1.4f}'.format(loss_test_hard))
    print('F1: {:1.3f}'.format(f1))
    print('F0.1: ', f01)
    print('F10: ', f10)
    print('Precision: {:1.3f}'.format(precision))
    print('Recall: {:1.3f}'.format(recall))


if __name__ == "__main__":
    data_folder = '../../data/multi-class-balanced-test/clean/'
    res_folder = '../../res/'
    dataset_files = ['5_train_corporate_messaging_mclass.csv',
                     '5_val_corporate_messaging_mclass.csv',
                     '5_test_corporate_messaging_mclass.csv']
    res_path = res_folder + '5_res_semisoft_corporate_messaging_mclass'

    # load and transform data
    data_params = {
        'dataset_files': dataset_files,
        'data_folder': data_folder,
        'text_column': 'text',
        'label_column': 'crowd_label',
        'min_df': 2,
        'max_features': None,
        'ngram_range': (1, 3)
    }
    data = load_data_mlp(**data_params)
    X_train_tfidf, y_train_soft, y_train_hard = data['train']
    X_val_tfidf, y_val_soft, y_val_hard = data['val']
    X_test_tfidf, y_test_hard = data['test']

    # transform data to tensors
    y_train_soft = Variable(torch.FloatTensor(y_train_soft))
    y_val_soft = Variable(torch.FloatTensor(y_val_soft))
    y_train_hard = Variable(torch.LongTensor(y_train_hard))
    y_val_hard = Variable(torch.LongTensor(y_val_hard))
    y_test_hard= Variable(torch.LongTensor(y_test_hard))
    X_train_tfidf = Variable(torch.Tensor(X_train_tfidf))
    X_val_tfidf = Variable(torch.Tensor(X_val_tfidf))
    X_test_tfidf = Variable(torch.Tensor(X_test_tfidf))

    # True if we evaluate model on test set
    # False if we do parameter search for model
    is_evaluation_experiment = True

    # parameters
    input_dim = X_train_tfidf.shape[1]
    output_dim = data['output_dim']

    if not is_evaluation_experiment:
        # init header of res file
        with open(res_path, 'w') as f:
            c = 'loss_val, pre_val, rec_val, acc_val, epoch, ece_val, f1_val, lr_rate, weight_decay, class_weight'
            f.write(c + '\n')

        epochs = 500
        for lr_rate in [0.1, 0.01, 0.001]:
            for weight_decay in [0.01, 0.001, 0.0001, 0.00001]:
                for class_weight in [[1, 3, 3], [1, 5, 5], [1, 7, 7], [1, 10, 10], [1, 12, 12]]:
                    class_weight = torch.Tensor(class_weight)
                    net_params = {
                        'lr_rate': lr_rate,
                        'weight_decay': weight_decay,
                        'class_weight': class_weight
                    }
                    print(net_params)
                    print('---------------------')
                    val_res = train_neural_net(net_params)

                    with open(res_path, 'a') as f:
                        s = ''
                        for i in val_res + [lr_rate, weight_decay, class_weight.numpy()]:
                            s += str(i) + ','
                        f.write(s[:-1] + '\n')

    # evaluate on test data
    if is_evaluation_experiment:
        net_params = {
            'lr_rate': 0.1,
            'weight_decay': 0.0001,
            'class_weight': torch.Tensor([1, 10, 10]),
            'epochs': 445
        }
        train_evaluate(net_params)
        print(net_params)
