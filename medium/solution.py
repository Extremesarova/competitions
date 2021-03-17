import json
import os
import pickle
from html.parser import HTMLParser

import lightgbm as lgb
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

PATH_TO_DATA = 'data'
PATH_TO_ADDITIONAL_DATA = 'additional_data'
AUTHOR = 'Artem_Ryblov'  # change here to <name>_<surname>
# it's a nice practice to define most of hyperparams here
SEED = 17
TITLE_NGRAMS = (1, 2)  # for tf-idf on titles
CONTENT_NGRAMS = (1, 2)  # for tf-idf on contents
MAX_FEATURES = 100000  # for tf-idf
LGB_TRAIN_ROUNDS = 60  # num. iteration to train LightGBM
LGB_NUM_LEAVES = 255  # max number of leaves in LightGBM trees
MEAN_TEST_TARGET = 4.33328  # what we got by submitting all zeros
RIDGE_WEIGHT = 0.6  # weight of Ridge predictions in a blend with LightGBM
LGB_WEIGHT = 0.4


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ' '.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')', ''))
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)
        return read_json_line(line=new_line)
    return result


def extract_features_and_write(path_to_data, path_to_save, inp_filename, is_train=True):
    titles = []
    contents = []
    dates = []
    authors = []
    features = ['content', 'published', 'title', 'author']
    prefix = 'train' if is_train else 'test'
    feature_files = [open(os.path.join(path_to_save,
                                       '{}_{}.txt'.format(prefix, feat)),
                          'w', encoding='utf-8')
                     for feat in features]

    with open(os.path.join(path_to_data, inp_filename), encoding='utf-8') as inp_json_file:

        for line in tqdm.tqdm(inp_json_file, desc=f"Reading {prefix} json files"):
            json_data = read_json_line(line)

            title = json_data['title'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\xa0', ' ')
            content = strip_tags(
                json_data['content'].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\xa0', ' '))
            published = json_data['published']
            author = json_data['meta_tags']['author']
            authors_name = json_data['meta_tags']['author']

            titles.append(title)
            contents.append(content)
            dates.append(published)
            authors.append(authors_name)

    dic = {'content': contents, 'published': dates, 'title': titles, 'author': authors}
    for feature in features:
        filename = prefix + "_" + feature + ".txt"
        with open(os.path.join(path_to_save, filename), 'wb') as fp:
            pickle.dump(dic[feature], fp)

    return titles, contents, dates, authors


# Time features
def add_time_features(dates):
    scaler = StandardScaler()
    hour = scaler.fit_transform(np.array([date.hour for date in dates]).reshape(-1, 1))
    weekday = scaler.fit_transform(np.array([date.weekday() for date in dates]).reshape(-1, 1))
    morning = scaler.fit_transform(((hour >= 7) & (hour <= 11)).astype('int').reshape(-1, 1))
    day = scaler.fit_transform(((hour >= 12) & (hour <= 18)).astype('int').reshape(-1, 1))
    evening = scaler.fit_transform(((hour >= 19) & (hour <= 23)).astype('int').reshape(-1, 1))
    night = scaler.fit_transform(((hour >= 0) & (hour <= 6)).astype('int').reshape(-1, 1))
    weekend_temp = np.array([date.weekday() for date in dates]).reshape(-1, 1)
    weekend = scaler.fit_transform(((weekend_temp >= 5) & (weekend_temp <= 6)).astype('int').reshape(-1, 1))

    feature_names = ['morning', 'day', 'evening', 'night', 'weekday']
    time_features = pd.DataFrame(list(zip(morning.flatten(),
                                          day.flatten(),
                                          evening.flatten(),
                                          night.flatten(),
                                          weekend.flatten())), columns=feature_names)
    sparse_time_features = csr_matrix(time_features.values)
    return sparse_time_features, feature_names


def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_ADDITIONAL_DATA,
                                                      'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')

    submission['log_recommends'] = prediction
    submission.to_csv(filename)


def main():
    train_titles, train_contents, train_dates, train_authors = extract_features_and_write(PATH_TO_DATA,
                                                                                          PATH_TO_ADDITIONAL_DATA,
                                                                                          'train.json',
                                                                                          is_train=True)
    test_titles, test_contents, test_dates, test_authors = extract_features_and_write(PATH_TO_DATA,
                                                                                      PATH_TO_ADDITIONAL_DATA,
                                                                                      'test.json',
                                                                                      is_train=False)
    print("Doing TF-IDF vectorization for articles")
    # Tf-Idf for article
    vectorizer_params = {'ngram_range': CONTENT_NGRAMS,
                         'max_features': MAX_FEATURES,
                         'tokenizer': lambda s: s.split(),
                         'stop_words': ENGLISH_STOP_WORDS,
                         }
    vectorizer_article = TfidfVectorizer(**vectorizer_params)
    X_train_article = vectorizer_article.fit_transform(train_contents)
    X_test_article = vectorizer_article.transform(test_contents)

    print("Doing TF-IDF vectorization for titles")
    # Tf-Idf for titles
    vectorizer_params = {'ngram_range': TITLE_NGRAMS,
                         'max_features': MAX_FEATURES,
                         'tokenizer': lambda s: s.split(),
                         'stop_words': ENGLISH_STOP_WORDS
                         }
    vectorizer_title = TfidfVectorizer(**vectorizer_params)
    X_train_title = vectorizer_title.fit_transform(train_titles)
    X_test_title = vectorizer_title.transform(test_titles)

    train_times = pd.to_datetime([date['$date'] for date in train_dates])
    test_times = pd.to_datetime([date['$date'] for date in test_dates])

    X_train_time_features_sparse, time_feature_names = add_time_features(train_times)
    X_test_time_features_sparse, _ = add_time_features(test_times)

    print("Doing bag of authors")
    # Bag of authors
    authors = np.unique(train_authors + test_authors)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(authors.reshape(-1, 1))
    enc.categories_
    X_train_author_sparse = enc.transform(np.array(train_authors).reshape(-1, 1)).toarray()
    X_test_author_sparse = enc.transform(np.array(test_authors).reshape(-1, 1)).toarray()

    # Additional features
    train_len = [len(article) for article in train_contents]
    test_len = [len(article) for article in test_contents]
    scaler = StandardScaler()

    X_train_len_sparse = scaler.fit_transform(np.array(train_len).reshape(-1, 1))
    X_test_len_sparse = scaler.fit_transform(np.array(test_len).reshape(-1, 1))

    X_train_sparse = hstack([X_train_article, X_train_title,
                             X_train_author_sparse,
                             X_train_time_features_sparse, X_train_len_sparse]).tocsr()

    X_test_sparse = hstack([X_test_article, X_test_title,
                            X_test_author_sparse,
                            X_test_time_features_sparse, X_test_len_sparse]).tocsr()

    train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'),
                               index_col='id')
    y_train = train_target['log_recommends'].values

    print("Doing Ridge")
    # alpha_values = np.logspace(-2, 2, 20)
    ridge = Ridge(random_state=42, alpha=0.01)
    # logit_grid_searcher = GridSearchCV(estimator=ridge, param_grid={'alpha': alpha_values}, scoring='neg_mean_absolute_error', n_jobs=4, cv=3, verbose=1)
    ridge.fit(X_train_sparse, y_train)
    # final_model = logit_grid_searcher.best_estimator_
    ridge_test_pred = ridge.predict(X_test_sparse)

    print("Doing Light GBM")
    lgb_x_train = lgb.Dataset(X_train_sparse.astype(np.float32),
                              label=np.log1p(y_train))

    param = {'num_leaves': LGB_NUM_LEAVES,
             'objective': 'mean_absolute_error',
             'metric': 'mae'}
    bst_lgb = lgb.train(param, lgb_x_train, LGB_TRAIN_ROUNDS, verbose_eval=5)
    lgb_test_pred = np.expm1(bst_lgb.predict(X_test_sparse.astype(np.float32)))

    mix_pred = LGB_WEIGHT * lgb_test_pred + RIDGE_WEIGHT * ridge_test_pred
    mix_test_pred_modif = mix_pred + MEAN_TEST_TARGET - y_train.mean()

    write_submission_file(mix_test_pred_modif, f'submission_alice_{AUTHOR}.csv')


if __name__ == "__main__":
    main()
