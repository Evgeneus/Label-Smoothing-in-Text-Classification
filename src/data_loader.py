import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data_hard(file_path, text_column='text', label_column='label', seed=None, test_size=0.2):
    df = pd.read_csv(file_path)

    # Train test split
    X = df[text_column].values
    y = df[label_column].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed,
                                                        test_size=test_size, shuffle=True)
    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=seed,
                                                      test_size=test_size, shuffle=True)

    tfidf = TfidfVectorizer(min_df=2, max_features=None,
                strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                stop_words=None, lowercase=False)
    tfidf.fit(X_train)

    X_train_tfidf = tfidf.transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    print("X_train_tfidf shape: {}".format(X_train_tfidf.shape))
    print("X_val_tfidf shape: {}".format(X_val_tfidf.shape))
    print("X_test_tfidf shape: {}".format(X_test_tfidf.shape))

    return X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test


def load_data_soft(file_path, text_column='text', label_column='label', seed=None, test_size=0.2):
    df = pd.read_csv(file_path)
    X = df[[text_column, 'conf_neg', 'conf_pos']]
    y = df[label_column].values
    # make soft target label
    for ind, label in enumerate(y):
        if label == 1:
            X.at[ind, 'conf_neg'] = 0.
        elif label == 0:
            X.at[ind, 'conf_pos'] = 0.

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed,
                                                        test_size=test_size, shuffle=True)
    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=seed,
                                                      test_size=test_size, shuffle=True)
    y_train_soft = X_train[['conf_neg', 'conf_pos']].values
    y_val_soft = X_val[['conf_neg', 'conf_pos']].values
    y_test_soft = X_test[['conf_neg', 'conf_pos']].values
    X_train = X_train[text_column].values
    X_val = X_val[text_column].values
    X_test = X_test[text_column].values

    tfidf = TfidfVectorizer(min_df=2, max_features=None,
                strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                stop_words=None, lowercase=False)

    tfidf.fit(X_train)
    X_train_tfidf = tfidf.transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)
    print("X_train_tfidf shape: {}".format(X_train_tfidf.shape))
    print("X_val_tfidf shape: {}".format(X_val_tfidf.shape))
    print("X_test_tfidf shape: {}".format(X_test_tfidf.shape))

    data = {
        'train': (X_train_tfidf, y_train_soft, y_train),
        'val': (X_val_tfidf, y_val_soft, y_val),
        'test': (X_test_tfidf, y_test_soft, y_test),
    }

    return data
