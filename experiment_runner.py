import math
from typing import List

import numpy as np
import pandas as pd
import statistics

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


K_FOLD = 'k-fold'
FSCORE = 'fscore'
RECALL = 'recall'
PRECISION = 'precision'
ACC = 'acc'
METRIC_FIELDS = [ACC, PRECISION, RECALL, FSCORE]

NUMBER_OF_PROJECTS = 'number_of_projects'
MIN = 'min'
MAX = 'max'
MEAN = 'mean'
MEDIAN = 'median'
NAN_FILES_BODIES = 'nan_files_bodies'
FILE_BODY = 'file_body'
PROJ_ID = 'proj_id'
LANGUAGE = 'language'


class ExperimentRunner:

    def __init__(self, data_filepath: str, verbose=False):
        self._df = load_data(data_filepath)
        self.verbose = verbose

    def train_eval_models(self, models_dict, metrics_filepath, n_splits=5):
        all_detail_metric_rows = []
        agg_rows = []
        for model_name, model_gen in models_dict.items():
            detail_metric_rows = self.train_eval_model(model_gen, n_splits=n_splits)
            all_detail_metric_rows += detail_metric_rows
            agg_rows.append(self._create_agg_metric_rows(model_name, detail_metric_rows))
        pd.DataFrame(all_detail_metric_rows).to_csv(f'{metrics_filepath}.detail.csv')
        pd.DataFrame(agg_rows).to_csv(f'{metrics_filepath}.agg.csv')

    def train_eval_model(self, model_gen, n_splits=5):
        detail_metric_rows = []
        for cv_idx, (train_x, train_y, test_x, test_y) in enumerate(gen_cv_subsets(self._df, n_splits)):
            if self.verbose:
                print(f'training {cv_idx+1}-Fold')
            model = model_gen()
            model.fit(train_x, train_y)
            y_pred = model.predict(test_x)
            precision, recall, fscore, _ = precision_recall_fscore_support(test_y, y_pred, average='macro', zero_division=0)
            metrics = {
                K_FOLD: cv_idx + 1,
                ACC: accuracy_score(test_y, y_pred),
                PRECISION: precision,
                RECALL: recall,
                FSCORE: fscore,
            }
            detail_metric_rows.append(metrics)
            if self.verbose:
                print(metrics)
        return detail_metric_rows

    def _create_agg_metric_rows(self, model_name, detail_metric_rows):
        agg_metrics = {'model-name': model_name}
        for field in METRIC_FIELDS:
            values = [row[field] for row in detail_metric_rows]
            agg_metrics[f'{field}-mean'] = statistics.mean(values)
            agg_metrics[f'{field}-stddev'] = statistics.stdev(values)
        return agg_metrics


def load_data(filepath):
    return pd.read_csv(filepath)

def gen_cv_subsets(df, n_splits):
    nan_items = stats_for_language(df)[NAN_FILES_BODIES].sum()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=1)
    x = np.empty(shape=(len(df) - nan_items), dtype=object)
    y = np.empty(shape=(len(df) - nan_items), dtype=object)
    groups = np.empty(shape=(len(df) - nan_items), dtype=int)
    index = 0
    for (language, proj_id), items in df.groupby([LANGUAGE, PROJ_ID]):
        proj_id = int(proj_id)
        for file_body in items[FILE_BODY]:
            if isinstance(file_body, float) and math.isnan(file_body):
                continue
            assert isinstance(language, str)
            assert isinstance(proj_id, int)
            assert isinstance(file_body, str), f'{type(file_body) = }'
            x[index] = file_body
            y[index] = language
            groups[index] = proj_id
            index += 1
    assert index == len(x)
    for train_idxs, test_idxs in cv.split(x, y, groups=groups):
        train_x = x[train_idxs]
        train_y = y[train_idxs]
        test_x = x[test_idxs]
        test_y = y[test_idxs]
        yield train_x, train_y, test_x, test_y


def stats_for_language(df: pd.DataFrame):
    # for (language, proj_id), item in df.groupby(['language', 'proj_id']):
    #     print()
    stats_rows = []
    for language in df[LANGUAGE].unique():
        lang_df = df.loc[df[LANGUAGE] == language]
        grouped = lang_df.groupby([PROJ_ID])
        files_per_project: List[int] = [len(items) for key, items in grouped]
        stats = {
            LANGUAGE: language,
            NUMBER_OF_PROJECTS: len(grouped),
            MIN: min(files_per_project),
            MAX: max(files_per_project),
            MEAN: statistics.mean(files_per_project),
            MEDIAN: statistics.median(files_per_project),
            NAN_FILES_BODIES: sum([sum([1 for file_body in items[FILE_BODY] if isinstance(file_body, float) and math.isnan(file_body)]) for key, items in grouped])
        }
        stats_rows.append(stats)
    return pd.DataFrame(stats_rows)


if __name__ == '__main__':
    # load_data('./data.csv')
    from basic_models import SvmWithVectorizer
    ExperimentRunner('./data.csv', verbose=True).train_eval_model(model_gen=SvmWithVectorizer)
