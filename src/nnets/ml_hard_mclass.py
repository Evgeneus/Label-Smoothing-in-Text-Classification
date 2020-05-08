from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression


from src.nnets.utils import ece_score, plot_reliability_diagram
from src.nnets.data_loader import load_data_ml_hard

seed = 2020


def train_clf():
    # parameters for grid search
    hyperparams_grid = {
        'LogisticRegression': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__class_weight': [{0: 1, 1: 1}, {0: 2, 1: 1}]
        }
    }
    param_vectorizer = {
        'tfidf__max_features': [15000, 20000, 25000, 30000, None],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'tfidf__min_df': [0, 1, 2, 3]
    }

    # define pipeline
    tfidf = TfidfVectorizer(min_df=2, max_features=None,
                            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                            stop_words=None, lowercase=False)
    model = LogisticRegression(random_state=seed)
    # model = LinearSVC(random_state=seed)
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', model)])
    model_name = model.__class__.__name__
    param_grid = {**param_vectorizer, **hyperparams_grid[model_name]}

    k = 5  # number of splits in CV
    num_classes = len(set(y_train_val))
    if num_classes == 2:
        scoring = 'f1'
    else:
        scoring = 'f1_macro'
    grid = GridSearchCV(pipeline, cv=k, param_grid=param_grid, scoring=scoring, n_jobs=4, verbose=1)
    grid.fit(X_train_val, y_train_val)

    score_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
    score_std = grid.cv_results_['std_test_score'][grid.best_index_]
    print('{} f1:{:1.3f} std: {:1.3f}, using {}'.format(model_name, score_mean, score_std, grid.best_params_))
    print('------------------------------------------------')


def train_evaluate(params):
    tfidf = TfidfVectorizer(min_df=params['min_df'], max_features=params['max_features'],
                            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                            ngram_range=params['ngram_range'], use_idf=1, smooth_idf=1, sublinear_tf=1,
                            stop_words=None, lowercase=False)

    tfidf.fit(X_train)
    # transform each document into a vectorv
    X_train_tfidf = tfidf.transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()
    print('Each of the %d documents is represented by %d features (TF-IDF score of unigrams and bigrams)' % (
        X_train_tfidf.shape))

    model = LogisticRegression(C=params['C'], penalty=params['penality'],
                               class_weight=params['class_weight'], random_state=seed)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)

    # check if binary or multi class classification
    num_classes = len(set(y_test))
    if num_classes == 2:
        average = 'binary'
    else:
        average = 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=average, beta=1)
    _, _, f01, _ = precision_recall_fscore_support(y_test, y_pred, average=average, beta=0.1)
    _, _, f10, _ = precision_recall_fscore_support(y_test, y_pred, average=average, beta=10)
    acc = accuracy_score(y_test, y_pred)
    ece = ece_score(y_test, y_pred_proba)

    plot_reliability_diagram(y_test, y_pred_proba, title_suffix='LogisticRegression (ECE={:1.4f})'.format(ece))
    print('*Evaluation on test data, {}*'.format(model.__class__.__name__))
    print('F1: ', f1)
    print('F0.1: ', f01)
    print('F10: ', f10)
    print('ECE: {:1.4f}'.format(ece))
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Accuracy: ', acc)


if __name__ == "__main__":
    data_folder = '../../data/datasets-with-crowd-votes/13.Amazon-isBook/clean/'
    dataset_files = ['train_MV.csv',
                     'amazon-isbook-val.csv',
                     'amazon-isbook-test.csv']
    # load and transform data
    data_params = {
        'dataset_files': dataset_files,
        'data_folder': data_folder,
        'text_column': 'text',
        'label_column_train': 'crowd_label',
        'label_column_val': 'gold_label',
        'label_column_test': 'gold_label',
    }
    # True if we evaluate model on test set
    # False if we do parameter search for model
    is_evaluation_experiment = True

    X_train, y_train, X_train_val, y_train_val, X_test, y_test = load_data_ml_hard(**data_params)

    if not is_evaluation_experiment:
        train_clf()

    if is_evaluation_experiment:
        params = {'C': 10,
                  'penality': 'l2',
                  'class_weight':  {0: 2, 1: 1},
                  'max_features': 25000,
                  'ngram_range': (1, 3),
                  'min_df': 0}
        train_evaluate(params)
