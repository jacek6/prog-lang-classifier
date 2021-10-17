# prog-lang-classifier

## My approach

1. Look into data
2. Prepare simple and easy weak classifier for given problem
3. prepare little framework/scripts to train and eval weak classifier - keep code generic in order to change classifier impl
4. calculate metrics, train test set split, make it all generic, in order to do it for many different models
5. implement more advanced models
6. compare advanced and simple one models
7. fine tune hyper-parameters of more advanced models with experiments
8. run experiments for whole night to find best hyper-parameters
9. evaluate models

## Notes

Placed data file in root folder in path `./data.csv`, Not pushing it to git, file is too large.

After look at the data:
 - `language` is a label which is going to be predicted
 - `proj_id` defines group of files
 - `file_id` is id of file
 - `file_body` contains actual content of file, which is going to be input for classifiction

Implement stratified K-Fold Crosvalidation which takes into account `proj_id`. Files with same `proj_id` are never splited into train and validatrion set.

Implement basic ML model based on SVM, just to test out if train and evaluation code works.

Evaluated basic ML model in order to have baseline. Futher more complex ML models are going to be compared to that.

By running:
```
 $ basic_models.py
```

We get evaluation of basic ML models in files: `stats-basic-models.agg.csv` and `stats-basic-models.detail.csv`.
 - `stats-basic-models.detail.csv` contains detailed metrics for each K-Fold cross-validation step for each model
 - `stats-basic-models.agg.csv` contains metrics aggregated over all crossvalidation steps, for each model

`stats-basic-models.agg.csv` contains mean and stddev and slice of it looks:
```
model-name	                    acc-mean	acc-stddev
basic	                        0.685148	0.005167
only lowercase	                0.703528	0.004569
only lowercase with 2-grams	    0.698803	0.004061
```

So it looks with SVM models, after this experiments, model which takes lowercase 1-grams performs slightly the best.