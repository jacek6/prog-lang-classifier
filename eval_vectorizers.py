import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from scipy.sparse import hstack

from experiment_runner import ExperimentRunner

BASIC_MODELS_GEN = {
    'basic': lambda: Model1(),
}

words_pattern = re.compile(r"(\b\w\w+\b)")
digit_pattern = re.compile(r"(\d+)")

def text_remove_alphanum(txt):
    txt = words_pattern.sub('A', txt)
    txt = digit_pattern.sub('D', txt)
    return txt

class Model1:

    def __init__(self):
        # self.vectorizer_no_aplhanum = CountVectorizer(lowercase=False,
        #                                               # ngram_range=(1, 1),  acc = 0.26
        #                                               # ngram_range=(1, 3),  acc=0.24
        #                                               ngram_range=(2, 2),  acc=0.25
        #                                   token_pattern=r"[^\s]"
        #                                   )
        self.vectorizer_no_aplhanum = CountVectorizer(lowercase=False, token_pattern=r"[^\s]")
        self.model = SVC()

    def fit(self, train_x, train_y):
        train_x_no_aplhanum = np.vectorize(text_remove_alphanum)(train_x)
        x_vector = self.vectorizer_no_aplhanum.fit_transform(train_x_no_aplhanum)

        print(train_x_no_aplhanum[0])
        print()
        print(self.vector_to_text(x_vector[0]))

        # self.model.fit(x_vector, train_y)

    def vector_to_text(self, v) -> str:
        return ' '.join(self.vectorizer_no_aplhanum.inverse_transform(v)[0].tolist())

    def predict(self, x):
        return self.model.predict(self.vectorizer_no_aplhanum.transform(x))


def eval_basic_models(verbose):
    r = ExperimentRunner('./data.csv', verbose=verbose)
    r.train_eval_models(BASIC_MODELS_GEN, './stats-model1-models')

if __name__ == '__main__':
    # TODO: dotenv
    eval_basic_models(verbose=True)