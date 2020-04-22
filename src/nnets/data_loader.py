import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data_ml_hard(dataset_files, data_folder, text_column='text', label_column='crowd_label'):
    # train data
    train_file = dataset_files[0]
    assert 'train' in train_file
    df_tr = pd.read_csv(data_folder + train_file)
    X_train = df_tr[text_column].values
    y_train_hard = df_tr[label_column].values

    # validation data
    val_file = dataset_files[1]
    assert 'val' in val_file
    df_val = pd.read_csv(data_folder + val_file)
    X_val = df_val[text_column].values
    y_val_hard = df_val[label_column].values

    # concatenate train val data for Cross Validation
    X_train_val = np.concatenate((X_train, X_val))
    y_train_val_hard = np.concatenate((y_train_hard, y_val_hard))

    # test data
    test_file = dataset_files[2]
    assert 'test' in test_file
    df_test = pd.read_csv(data_folder + test_file)
    X_test = df_test[text_column].values
    y_test_hard = df_test['gold_label'].values

    return X_train, y_train_hard, X_train_val, y_train_val_hard, X_test, y_test_hard


def load_data_soft(dataset_files, data_folder, text_column='text', label_column='crowd_label', min_df=2, max_features=None, ngram_range=(1, 3)):
    # train data
    train_file = dataset_files[0]
    assert 'train' in train_file
    df_tr = pd.read_csv(data_folder + train_file)
    X_train = df_tr[text_column].values
    y_train_hard = df_tr[label_column].values
    # # make semi soft target label
    # labels_set = set(df_tr[label_column].unique())
    # for ind, label in enumerate(y_train_hard):
    #     for _label in labels_set:
    #         if _label == label: continue
    #         df_tr.at[ind, 'conf{}'.format(_label)] = 0
    y_train_soft = df_tr.iloc[:, 2:].values

    # validation data
    val_file = dataset_files[1]
    assert 'val' in val_file
    df_val = pd.read_csv(data_folder + val_file)
    X_val = df_val[text_column].values
    y_val_hard = df_val[label_column].values
    labels_set = set(df_val[label_column].unique())
    # # make semi soft target label
    # for ind, label in enumerate(y_val_hard):
    #     for _label in labels_set:
    #         if _label == label: continue
    #         df_val.at[ind, 'conf{}'.format(_label)] = 0
    y_val_soft = df_val.iloc[:, 2:].values

    # test data
    test_file = dataset_files[2]
    assert 'test' in test_file
    df_test = pd.read_csv(data_folder + test_file)
    X_test = df_test[text_column].values
    y_test_hard = df_test['gold_label'].values

    # compute tfidf features
    tfidf = TfidfVectorizer(min_df=min_df, max_features=max_features,
                strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                stop_words=None, lowercase=False)

    tfidf.fit(X_train)
    X_train_tfidf = tfidf.transform(X_train).todense()
    X_val_tfidf = tfidf.transform(X_val).todense()
    X_test_tfidf = tfidf.transform(X_test).todense()
    print("X_train_tfidf shape: {}".format(X_train_tfidf.shape))
    print("X_val_tfidf shape: {}".format(X_val_tfidf.shape))
    print("X_test_tfidf shape: {}".format(X_test_tfidf.shape))

    data = {
        'train': (X_train_tfidf, y_train_soft, y_train_hard),
        'val': (X_val_tfidf, y_val_soft, y_val_hard),
        'test': (X_test_tfidf, y_test_hard),
        'output_dim': len(labels_set)
    }

    return data
