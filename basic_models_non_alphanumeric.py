import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from experiment_runner import ExperimentRunner

BASIC_MODELS_GEN = {
    'basic': lambda: SvmWithVectorizer(vectorizer_kwargs=dict(token_pattern=r"[^\s]")),
    'only lowercase': lambda: SvmWithVectorizer(vectorizer_kwargs=dict(lowercase=False, max_df=1.0, ngram_range=(1, 1), token_pattern=r"[^\s]")),
    'only lowercase with 2-grams': lambda: SvmWithVectorizer(vectorizer_kwargs=dict(lowercase=False, max_df=1.0, ngram_range=(1, 2), token_pattern=r"[^\s]")),
    'only lowercase with 3-grams': lambda: SvmWithVectorizer(vectorizer_kwargs=dict(lowercase=False, max_df=1.0, ngram_range=(1, 3), token_pattern=r"[^\s]")),
}

words_pattern = re.compile(r"(\b\w\w+\b)")
digit_pattern = re.compile(r"(\d+)")


def text_remove_alphanum(txt):
    txt = words_pattern.sub('A', txt)
    txt = digit_pattern.sub('D', txt)
    return txt


class SvmWithVectorizer:

    def __init__(self, vectorizer_kwargs=None, svc_kwargs=None):
        vectorizer_kwargs = vectorizer_kwargs or {}
        svc_kwargs = svc_kwargs or {}
        self.vectorizer = CountVectorizer(**vectorizer_kwargs)
        self.model = SVC(**svc_kwargs)

    def fit(self, train_x, train_y):
        train_x_no_aplhanum = np.vectorize(text_remove_alphanum)(train_x)
        x_vector = self.vectorizer.fit_transform(train_x_no_aplhanum)
        self.model.fit(x_vector, train_y)

    def predict(self, x):
        x_no_aplhanum = np.vectorize(text_remove_alphanum)(x)
        return self.model.predict(self.vectorizer.transform(x_no_aplhanum))


def eval_basic_models(verbose):
    r = ExperimentRunner('./data.csv', verbose=verbose)
    r.train_eval_models(BASIC_MODELS_GEN, './stats-basic-models-non-alphanumeric')


if __name__ == '__main__':
    # TODO: dotenv
    eval_basic_models(verbose=True)
