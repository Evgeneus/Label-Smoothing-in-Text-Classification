import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm
from uuid import uuid4

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

## PyTorch Transformer
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig

from sklearn.metrics import precision_recall_fscore_support


config = RobertaConfig.from_pretrained('roberta-base')
config.num_labels = 2

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification(config)

## Feature Preparation
def prepare_features(seq_1, max_seq_length = 300,
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

# msg = "My dog is cute!"
# print(prepare_features(msg))

## Dataset Loader Classes
class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        text = self.data.text[index]
        X, _ = prepare_features(text)
        # TODO:
        y = self.data.crowd_label[index]
        return X, y

    def __len__(self):
        return self.len


# -----------------------------
data_folder = '../../data/binary/raw/'

# train_size = 0.8
# train_dataset = dataset.sample(frac=train_size,random_state=200).reset_index(drop=True)
# test_dataset = dataset.drop(train_dataset.index).reset_index(drop=True)

train_file = '1_raw_train_gop_sentiment_binary.csv'
val_file = '1_raw_val_gop_sentiment_binary.csv'
test_file = '1_raw_test_gop_sentiment_binary.csv'
train_dataset = pd.read_csv(data_folder + train_file)
val_dataset = pd.read_csv(data_folder + val_file)
test_dataset = pd.read_csv(data_folder + test_file)

print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(val_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))
training_set = Intents(train_dataset)
validating_set = Intents(val_dataset)
testing_set = Intents(test_dataset)


## Training Params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model = model.cuda()

# Parameters
params = {'batch_size': 30,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 1}

training_loader = DataLoader(training_set, **params)
validating_loader = DataLoader(validating_set, **params)
testing_loader = DataLoader(testing_set, **params)

class_weight = torch.Tensor([1, 12])
if torch.cuda.is_available():
  class_weight = class_weight.cuda()
loss_function = nn.CrossEntropyLoss(weight=class_weight)
# learning_rate = 1e-05
learning_rate = 1e-03
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

## Test Forward Pass
# inp = training_set.__getitem__(0)[0].cuda()
# inp = training_set.__getitem__(0)[0]
# output = model(inp)[0]
# print(output.shape)

max_epochs = 3
model = model.train()
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

        if i % 10 == 0:
            model.eval()
            y_pred = []
            y_val_hard = []
            correct = 0
            total = 0
            for sent, label in validating_loader:
                y_val_hard.append(label.item())
                sent = sent.squeeze(0)
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                output = model.forward(sent)[0]
                _, predicted = torch.max(output.data, 1)
                y_pred.append(predicted.item())
                total += label.size(0)
            model.train()
            pre_val, rec_val, f1_val, _ = precision_recall_fscore_support(y_val_hard, y_pred, average='binary', beta=1)
            print('Iteration: {}. Validation, Loss: {}. F1: {:1.3f}, Precision: {:1.3f}, Recall: {:1.3f}'.
                  format(i, loss.item(), f1_val, pre_val, rec_val))

# torch.save(model.state_dict(), 'drive/My Drive/Datasets/roberta_state_dict_'+ str(uuid4())+'.pth')
