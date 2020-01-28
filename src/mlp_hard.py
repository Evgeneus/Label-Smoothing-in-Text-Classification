import torch
from torch.autograd import Variable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


seed = 2020

df = pd.read_csv('../data/GOP_REL_ONLY_cleaned_stem.csv')
text_column = 'text'
label_column = 'label'

# Train test split
X = df[text_column].values
y = df[label_column].values
test_size = 0.3

torch.manual_seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed,
                                                    test_size=test_size, shuffle=True)

tfidf = TfidfVectorizer(min_df=2, max_features=None,
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words=None, lowercase=False)

tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("X_train_tfidf shape: {}".format(X_train_tfidf.shape))
print("X_test_tfidf shape: {}".format(X_test_tfidf.shape))


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


epochs = 5000
input_dim = X_train_tfidf.shape[1]
output_dim = 2
lr_rate = 0.05

model = LogisticRegression(input_dim, output_dim)

criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 2]))  # computes softmax and then the cross entropy

optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=0.0001)

y_train = Variable(torch.LongTensor(y_train))
y_test = Variable(torch.LongTensor(y_test))
X_train_tfidf = Variable(torch.Tensor(tfidf.transform(X_train).todense()))
X_test_tfidf = Variable(torch.Tensor(tfidf.transform(X_test).todense()))

for epoch in range(int(epochs)):
    optimizer.zero_grad()
    outputs = model(X_train_tfidf)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(epoch)
        model.eval()
        with torch.no_grad():
            # Get predictions from the maximum value
            outputs_test = torch.sigmoid(model(X_test_tfidf))
            _, y_pred = torch.max(outputs_test.data, 1)
            loss_test = criterion(outputs_test, y_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', beta=1)
        avg_precision = average_precision_score(y_test.numpy(), outputs_test.data.numpy()[:, 1])
        roc_auc = roc_auc_score(y_test.numpy(), outputs_test.data.numpy()[:, 1])
        model.train()

        print('*Evaluation on test data, epoch {}*'.format(epoch))
        print('Avg Precision: {:1.3f}'.format(avg_precision))
        print('ROC AUC: {:1.3f}'.format(roc_auc))
        print('f1: {:1.3f}'.format(f1))
        print('Precision: {:1.3f}'.format(precision))
        print('Recall: {:1.3f}'.format(recall))
        print('train loss: ', loss)
        print('test  loss: ', loss_test)
        print('------------')






