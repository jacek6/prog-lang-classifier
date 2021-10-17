from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from experiment_runner import ExperimentRunner

BASIC_MODELS_GEN = {
    'basic': lambda: SvmWithVectorizer(),
    'only lowercase': lambda: SvmWithVectorizer(vectorizer_kwargs=dict(lowercase=False, max_df=0.6, ngram_range=(1, 1))),
    'only lowercase with 2-grams': lambda: SvmWithVectorizer(vectorizer_kwargs=dict(lowercase=False, max_df=0.6, ngram_range=(1, 2))),
}

class SvmWithVectorizer:

    def __init__(self, vectorizer_kwargs=None, svc_kwargs=None):
        vectorizer_kwargs = vectorizer_kwargs or {}
        svc_kwargs = svc_kwargs or {}
        self.vectorizer = CountVectorizer(**vectorizer_kwargs)
        self.model = SVC(**svc_kwargs)

    def fit(self, train_x, train_y):
        x_vector = self.vectorizer.fit_transform(train_x)
        self.model.fit(x_vector, train_y)

    def predict(self, x):
        return self.model.predict(self.vectorizer.transform(x))

def eval_basic_models(verbose):
    r = ExperimentRunner('./data.csv', verbose=verbose)
    r.train_eval_models(BASIC_MODELS_GEN, './stats-basic-models')

if __name__ == '__main__':
    # TODO: dotenv
    eval_basic_models(verbose=True)