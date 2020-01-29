import torch
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from src.model import MLP1
from src.data_loader import load_data_hard

seed = 2020
torch.manual_seed(seed)


if __name__ == "__main__":
    file_path = '../data/GOP_REL_ONLY_cleaned_stem.csv'
    text_column = 'text'
    label_column = 'label'

    # load and transform data
    X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test = load_data_hard(file_path, text_column, label_column, seed)
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
    class_weight = torch.Tensor([1, 2])

    # define NNet and training process
    model = MLP1(input_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)  # computes softmax and then the cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tfidf)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # evaluate model on val data
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tfidf)
            loss_val = criterion(outputs_val, y_val)
        print('Epoch {}. Train Loss : {:1.3f} | Val Loss: {:1.3f}'.format(epoch, loss, loss_val))

    # evaluate on test data
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_tfidf)
        loss_test = criterion(outputs_test, y_test).item()
        # Get predictions from the maximum value
        _, y_pred = torch.max(torch.sigmoid(outputs_test).data, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', beta=1)
    avg_precision = average_precision_score(y_test.numpy(), outputs_test.data.numpy()[:, 1])
    roc_auc = roc_auc_score(y_test.numpy(), outputs_test.data.numpy()[:, 1])

    print('------------')
    print('*Evaluation on test data, epoch {}*'.format(epoch))
    print('Test Loss: {:1.4}'.format(loss_test))
    print('F1: {:1.3f}'.format(f1))
    print('Avg Precision: {:1.3f}'.format(avg_precision))
    print('Precision: {:1.3f}'.format(precision))
    print('Recall: {:1.3f}'.format(recall))






