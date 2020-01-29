import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data_hard(file_path, text_column='text', label_column='label', seed=None):
    # df = pd.read_csv('../data/GOP_REL_ONLY_cleaned_stem.csv')
    # text_column = 'text'
    # label_column = 'label'
    df = pd.read_csv(file_path)

    # Train test split
    X = df[text_column].values
    y = df[label_column].values
    test_size = 0.3

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

    return X_train_tfidf, y_train, X_test_tfidf, y_test

