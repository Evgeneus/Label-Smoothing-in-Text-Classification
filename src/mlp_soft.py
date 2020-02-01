import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

from src.model import MLP1
from src.data_loader import load_data_soft
from src.utils import CrossEntropyLossSoft
from src.utils import ece_score, plot_reliability_diagram

seed = 2020
torch.manual_seed(seed)


if __name__ == "__main__":
    file_path = '../data/GOP_REL_ONLY_cleaned_stem_confmin0.6.csv'
    text_column = 'text'
    label_column = 'label'

    # load and transform data
    data = load_data_soft(file_path, text_column, label_column, seed)
    X_train_tfidf, y_train_soft, y_train = data['train']
    X_val_tfidf, y_val_soft, y_val = data['val']
    X_test_tfidf, y_test_soft, y_test = data['test']

    # transform data to tensors
    y_train_soft = Variable(torch.FloatTensor(y_train_soft))
    y_val_soft = Variable(torch.FloatTensor(y_val_soft))
    y_test_soft = Variable(torch.FloatTensor(y_test_soft))
    y_train = Variable(torch.LongTensor(y_train))
    y_val = Variable(torch.LongTensor(y_val))
    y_test = Variable(torch.LongTensor(y_test))
    X_train_tfidf = Variable(torch.Tensor(X_train_tfidf.todense()))
    X_val_tfidf = Variable(torch.Tensor(X_val_tfidf.todense()))
    X_test_tfidf = Variable(torch.Tensor(X_test_tfidf.todense()))

    # parameters
    epochs = 150
    input_dim = X_train_tfidf.shape[1]
    output_dim = 2
    lr_rate = 0.05
    weight_decay = 0.0001
    # compute class weight
    # class_sample_count = np.unique(y_train, return_counts=True)[1]
    # class_weight_balanced = torch.FloatTensor(1. / class_sample_count)
    class_weight_balanced = torch.Tensor([1, 10])

    # define NNet and training process
    model = MLP1(input_dim, output_dim)
    criterion = CrossEntropyLossSoft(class_weight_balanced)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    for epoch in range(1, int(epochs)+1):
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
            loss_val = criterion(outputs_val, y_val_soft)
            ece_val = ece_score(y_val.numpy(), torch.sigmoid(outputs_val)[:, 1].numpy())
            _, y_pred = torch.max(torch.sigmoid(outputs_val).data, 1)
            pre_val, rec_val, f1_val, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', beta=1)
            acc_val = accuracy_score(y_val, y_pred)
        print('Epoch {}. Train Loss : {:1.3f} | Val Loss: {:1.3f} | Val ECE: {:1.3f} | Val F1: {:1.4f} | Val Acc: {:1.3f}'
              'Val P: {:1.3f} | Val R: {:1.3f}'.format(epoch, loss, loss_val, ece_val, f1_val, pre_val, rec_val, acc_val))

    # evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tfidf)
        loss_test_soft = criterion(outputs_test, y_test_soft)
        CrossEntropyLossHard = torch.nn.CrossEntropyLoss()
        loss_test_hard = CrossEntropyLossHard(outputs_test, y_test).item()
        # Get predictions from the maximum value
        _, y_pred = torch.max(torch.sigmoid(outputs_test).data, 1)
    ece_test = ece_score(y_test.numpy(), torch.sigmoid(outputs_test)[:, 1].numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', beta=1)
    avg_precision = average_precision_score(y_test.numpy(), outputs_test.data.numpy()[:, 1])

    plot_reliability_diagram(y_test.numpy(), torch.sigmoid(outputs_test)[:, 1].numpy())
    print('------------')
    print('*Evaluation on test data, epoch {}*'.format(epoch))
    print('Test ECE: {:1.4f}'.format(ece_test))
    print('Test Loss Soft: {:1.4f}'.format(loss_test_soft))
    print('Test Loss Hard: {:1.4f}'.format(loss_test_hard))
    print('F1: {:1.3f}'.format(f1))
    print('Avg Precision: {:1.3f}'.format(avg_precision))
    print('Precision: {:1.3f}'.format(precision))
    print('Recall: {:1.3f}'.format(recall))







