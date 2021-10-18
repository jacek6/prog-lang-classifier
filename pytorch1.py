import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

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
        special_chars = "!@#$%^&*(){}[];'\|,.//\""
        special_chars_enclosed = [f'\\{ch}' for ch in special_chars]
        self.vectorizer = CountVectorizer(lowercase=False, ngram_range=(1, 1),
                                          # token_pattern=r"[\b\w\w+\b]|" + f"[{'|'.join(special_chars_enclosed)}]"
                                          token_pattern=r"(\b\w\w+\b)|\{"
                                          )

    def fit(self, train_x, train_y):
        v = self.vectorizer.fit_transform(train_x)
        print(train_x[0])
        print()
        print(text_remove_alphanum(train_x[0]))
        print()

        print(self.vector_to_text(v[0]))
        print(self.vector_to_text(v[1]))

    def predict(self, x):
        pass

    def vector_to_text(self, v) -> str:
        return ' '.join(self.vectorizer.inverse_transform(v)[0].tolist())

def eval_basic_models(verbose):
    r = ExperimentRunner('./data.csv', verbose=verbose)
    r.train_eval_models(BASIC_MODELS_GEN, './stats-model1-models')

if __name__ == '__main__':
    # TODO: dotenv
    eval_basic_models(verbose=True)