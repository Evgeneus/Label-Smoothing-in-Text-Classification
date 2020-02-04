from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data_loader import load_data_ml_hard
from src.utils import ece_score, plot_reliability_diagram

seed = 2020


def train_clf():
    # parameters for grid search
    hyperparams_grid = {
        'LogisticRegression': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2},
                                  {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 7},
                                  {0: 1, 1: 10}, {0: 1, 1: 12}]
        },
        'LinearSVC': {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l2'],
            'clf__loss': ['hinge', 'squared_hinge'],
            'clf__class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2},
                                  {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 7},
                                  {0: 1, 1: 10}, {0: 1, 1: 12}]
        }
    }
    param_vectorizer = {
        'tfidf__max_features': [15000, 20000, 25000, 30000],
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
    grid = GridSearchCV(pipeline, cv=k, param_grid=param_grid, scoring='f1', n_jobs=4, verbose=1)
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
    # model = LinearSVC(C=params['C'], penalty=params['penality'],
    #                   class_weight=params['class_weight'], random_state=seed)
    # from sklearn.calibration import CalibratedClassifierCV
    # model = CalibratedClassifierCV(model)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', beta=1)
    acc = accuracy_score(y_test, y_pred)
    ece = ece_score(y_test, y_pred_proba)

    plot_reliability_diagram(y_test, y_pred_proba, title_suffix='LogisticRegression (ECE={:1.4f})'.format(ece))
    print('*Evaluation on test data, {}*'.format(model.__class__.__name__))
    print('F1: ', f1)
    print('ECE: {:1.4f}'.format(ece))
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Accuracy: ', acc)


if __name__ == "__main__":
    data_folder = '../data/'
    dataset_file = 'GOP_REL_ONLY_cleaned_stem_confmin0.6.csv'
    dataset_path = data_folder + dataset_file
    text_column, label_column = 'text', 'label'

    # True if we evaluate model on test set
    # False if we do parameter search for model
    is_evaluation_experiment = True

    # load and transform data
    X_train, y_train, X_train_val, y_train_val, X_test, y_test = load_data_ml_hard(dataset_path, text_column,
                                                                                  label_column, seed)
    if not is_evaluation_experiment:
        train_clf()

    if is_evaluation_experiment:
        params = {'C': 1,
                  'penality': 'l2',
                  'class_weight': {0: 1, 1: 3},
                  'max_features': None,
                  'ngram_range': (1, 3),
                  'min_df': 2}
        train_evaluate(params)
