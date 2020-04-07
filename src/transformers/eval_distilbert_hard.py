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

from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

## PyTorch Transformer
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertConfig
from sklearn.metrics import precision_recall_fscore_support


seed = 2020
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


config = DistilBertConfig.from_pretrained('distilbert-base-cased')
config.num_labels = 3

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertForSequenceClassification(config)


## Feature Preparation
def prepare_features(seq_1, max_seq_length = 512,
             zero_pad = True, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask


## Dataset Loader Classes
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


def ece_score(y_true, y_prob, n_bins=10):
    ece = ECE(n_bins)
    ece_val = ece.measure(y_prob, y_true)

    return ece_val


def compute_val():
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
        # compute and plot ECE
        ece_val = ece_score(np.array(y_val_hard), np.array(output_prob_val))
        n_bins = 10
        title_suffix = ''
        diagram = ReliabilityDiagram(n_bins)
        diagram.plot(np.array(output_prob_val), np.array(y_val_hard), title_suffix)
        # plt.savefig(title_suffix + '.pdf')

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
            'Iteration: {}. Train Loss: {:1.5f}. Test Loss: {:1.5f}, F1: {:1.3f}, ECE: {:1.3f}, Precision: {:1.3f}, Recall: {:1.3f}'.
            format(i, loss.item(), loss_val, f1_val, ece_val, pre_val, rec_val))
        # print to result file
        with open(res_path, 'w') as f:
            c = 'epoch, iter, loss_train, loss_test, pre_test, rec_test, f01_test, f1_test, f10_test, ece_test'
            f.write(c + '\n')
            res_i = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(epoch, i, loss.item(), loss_val, pre_val, rec_val,
                                                                      f01_val, f1_val, f10_val, ece_val)
            f.write(res_i)


if __name__ == "__main__":
    data_folder = '../../data/multi-class-balanced-test/tobert/'
    log_file = "1-hard-gop_sentiment_binary-lr10-6"
    res_path = '../../res/' + 'test_' + log_file + '.csv'

    train_file = '8_bert_train_drug_relation_mclass.csv'
    test_file = '8_bert_test_drug_relation_mclass.csv'
    train_dataset = pd.read_csv(data_folder + train_file)
    test_dataset = pd.read_csv(data_folder + test_file)

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    training_set = DataLoaderHard(train_dataset)
    testing_set = DataLoaderHard(test_dataset)

    # Sampler
    target = train_dataset.crowd_label.values
    print('target train 0/1: {}/{}'.format(
        len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    ## Training Params
    if torch.cuda.is_available():
        model = model.cuda()

    # Parameters
    # 'batch_size': 35, max_seq_length = 512 for GPU with 16GB memory
    params = {'batch_size': 32,
              # 'shuffle': True,
              'drop_last': False,
              'num_workers': 4,
              'sampler': sampler}

    params_val = {'batch_size': 1,
              'shuffle': True,
              'drop_last': False,
              'num_workers': 1}

    training_loader = DataLoader(training_set, **params)
    validating_loader = DataLoader(testing_set, **params_val)

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 1e-06
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    model = model.train()
    iter_eval = 0
    epoch_eval = 88
    max_epochs = epoch_eval + 1
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

            if (epoch == epoch_eval) and (i == iter_eval):
              compute_val()
              print('Model Evaluated!')
              torch.save(model.state_dict(), 'test_{}_state_dict.pth'.format(log_file))
              exit(0)
