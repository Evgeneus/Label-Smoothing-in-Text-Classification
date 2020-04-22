# !pip install transformers
# !pip install sklearn
# !pip install netcal
#
# ## Check if Cuda is Available
# print(torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#   print('GPU Properties:   ', torch.cuda.get_device_properties(0))
#   print('Memory Usage:')
#   print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#   print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
#
#
# ## Mount Drive into Colab
# from google.colab import drive
# drive.mount('/content/drive')
# # ////////////////////////////////////////////////////////////

import pandas as pd
import numpy as np
from tqdm import tqdm
import random

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

## PyTorch Transformer
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertConfig

from sklearn.metrics import precision_recall_fscore_support
from netcal.metrics import ECE

# setup random seed
seed = 2020
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class CrossEntropyLossSoft(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLossSoft, self).__init__()
        self.weight = weight

    def forward(self, pred, soft_targets):
        logsoftmax = nn.LogSoftmax()
        if self.weight is not None:
            return torch.mean(torch.sum(- soft_targets * self.weight * logsoftmax(pred), 1))
        else:
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


class DataLoaderHard(Dataset):

    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        text = self.data.text[index]
        X, _ = prepare_features(text)
        y = self.data.crowd_label[index]

        return X, y

    def __len__(self):
        return self.len


class DataLoaderSmoothing(Dataset):

    def __init__(self, dataframe, alpha):
        self.len = len(dataframe)
        self.data = dataframe
        self.alpha = alpha
        self.num_classes = len(self.data.crowd_label.unique())

    def __getitem__(self, index):
        text = self.data.text[index]
        X, _ = prepare_features(text)
        y = self.data.loc[index].iloc[2:].values.astype(float)
        crowd_label = self.data.loc[index].crowd_label
        for ind in range(len(y)):
            if ind == crowd_label:
                y[ind] = 1 - self.alpha + self.alpha / self.num_classes
                pass
            else:
                y[ind] = self.alpha / self.num_classes

        return X, y


# feature preparation for BERT
def prepare_features(seq_1, max_seq_length=512, zero_pad=True,
                     include_CLS_token = True, include_SEP_token = True):
    # Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    # Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    # Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    # Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Input Mask
    input_mask = [1] * len(input_ids)
    # Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask


def ece_score(y_true, y_prob, n_bins=10):
    ece = ECE(n_bins)
    ece_val = ece.measure(y_prob, y_true)

    return ece_val


def compute_val():
    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        y_pred = []
        output_prob_val = []
        output_logits_val = []
        y_val_hard = []
        for sent, label in validating_loader:
            y_val_hard.append(label.item())
            sent = sent.squeeze(0)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            logit, predicted = torch.max(output.data, 1)
            output_logits_val.append(output[0].cpu().tolist())
            output_prob_val.append(torch.sigmoid(output[0]).cpu().tolist())
            y_pred.append(predicted.item())
        loss_val = loss_function(torch.Tensor(output_logits_val), torch.LongTensor(y_val_hard)).item()
        model.train()
        ece_val = ece_score(np.array(y_val_hard), np.array(output_prob_val))

        # check if binary or multi class classification
        num_classes = len(set(y_val_hard))
        if num_classes == 2:
            average = 'binary'
        else:
            average = 'macro'
        pre_val, rec_val, f1_val, _ = precision_recall_fscore_support(y_val_hard, y_pred, average=average, beta=1)
        _, _, f01_val, _ = precision_recall_fscore_support(y_val_hard, y_pred, average=average, beta=0.1)
        _, _, f10_val, _ = precision_recall_fscore_support(y_val_hard, y_pred, average=average, beta=10)
        print(
            'Iteration: {}. Train Loss: {:1.5f}. Val Loss: {:1.5f}, F1: {:1.3f}, ECE: {:1.3f}, Precision: {:1.3f}, Recall: {:1.3f}'.
            format(i, loss.item(), loss_val, f1_val, ece_val, pre_val, rec_val))
        # print to result file
        with open(res_path, 'a') as f:
            res_i = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(epoch, i, loss.item(), loss_val, pre_val, rec_val,
                                                                      f01_val, f1_val, f10_val, ece_val)
            f.write(res_i)


if __name__ == "__main__":
    # PARAMETERS
    logfile_name = "5-lsmooth-gop_sentiment-lr10-6-1&1cw-a01.csv"
    train_file = '5_bert_train_corporate_messaging_mclass.csv'
    val_file = '5_bert_val_corporate_messaging_mclass.csv'
    num_labels = 3
    class_weight = torch.Tensor([1, 1, 1])
    batch_size = 32
    learning_rate = 1e-06
    max_epochs = 100
    alpha = 0.1  # smoothing parameters for true label
    # /PARAMETERS

    # create log file
    data_folder = '../../data/multi-class-balanced-test/tobert/'
    res_path = '../../res/'
    res_path += logfile_name
    with open(res_path, 'w') as f:
        c = 'epoch, iter, loss_train, loss_val, pre_val, rec_val, f01_val, f1_val, f10_val, ece_val'
        f.write(c + '\n')

    # configure DistilBERT model
    config = DistilBertConfig.from_pretrained('distilbert-base-cased')
    config.num_labels = num_labels
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    model = DistilBertForSequenceClassification(config)
    # load model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # load datasets
    train_dataset = pd.read_csv(data_folder + train_file)
    val_dataset = pd.read_csv(data_folder + val_file)
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))
    training_set = DataLoaderSmoothing(train_dataset, alpha)
    validating_set = DataLoaderHard(val_dataset)

    # initialize batch sampler
    target = train_dataset.crowd_label.values
    print('target train 0/1: {}/{}'.format(
        len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    # parameters: 'batch_size': 35, max_seq_length = 512 for GPU with 16GB memory
    params_train = {'batch_size': batch_size,
                    'drop_last': False,
                    'num_workers': 4,
                    'sampler': sampler}
    params_val = {'batch_size': 1,
                  'shuffle': True,
                  'drop_last': False,
                  'num_workers': 4}
    training_loader = DataLoader(training_set, **params_train)
    validating_loader = DataLoader(validating_set, **params_val)

    # initialize loss function and optimizer
    if torch.cuda.is_available():
      class_weight = class_weight.cuda()
    loss_function = CrossEntropyLossSoft(weight=class_weight)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    model = model.train()

    # train model
    for epoch in tqdm(range(max_epochs)):
        print("EPOCH -- {}".format(epoch))
        for i, (sent, label) in enumerate(training_loader):
            optimizer.zero_grad()
            sent = sent.squeeze(1)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
              compute_val()
