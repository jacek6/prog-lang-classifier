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
10. document results

## Notes

Placed data file in root folder in path `./data.csv`, Not pushing it to git, file is too large.

After look at the data:
 - `language` is a label which is going to be predicted
 - `proj_id` defines group of files
 - `file_id` is id of file
 - `file_body` contains actual content of file, which is going to be input for classifiction

Implement stratified K-Fold Crosvalidation which takes into account `proj_id`. Files with same `proj_id` are never splited into train and validation set.

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
model-name	                        acc-mean	       acc-stddev
basic	                            0.685148	       0.005167
only lowercase	                    0.703528           0.004569
only lowercase with 2-grams	        0.698803	       0.004061
```

So it looks with SVM models, after this experiments, model which takes lowercase 1-grams performs slightly the best (the middle row).

### Data preparation
Text of file bodies need to be converted to vector for many ML methods.

Let's look by example how text is processed. Below we have example content of some file body from given data.
```
#include <stdio.h>
#include <stdlib.h>

typedef struct floatList {
    float *list;
    int   size;
} *FloatList;

int floatcmp( const void *a, const void *b) {
    if (*(const float *)a < *(const float *)b) return -1;
    else return *(const float *)a > *(const float *)b;
}

float median( FloatList fl )
{
    qsort( fl->list, fl->size, sizeof(float), floatcmp);
    return 0.5 * ( fl->list[fl->size/2] + fl->list[(fl->size-1)/2]);
}

int main()
{
    static float floats1[] = { 5.1, 2.6, 6.2, 8.8, 4.6, 4.1 };
    static struct floatList flist1 = { floats1, sizeof(floats1)/sizeof(float) };

    static float floats2[] = { 5.1, 2.6, 8.8, 4.6, 4.1 };
    static struct floatList flist2 = { floats2, sizeof(floats2)/sizeof(float) };

    printf("flist1 median is %7.2f\n", median(&flist1)); /* 4.85 */
    printf("flist2 median is %7.2f\n", median(&flist2)); /* 4.60 */
    return 0;
}
```

For that text, we may apply some simply method for text vectorizing, for example `CountVectorizer` from `sklearn.feature_extraction.text`.

With such simple method, if we vectorize and 'un-vectorize' we get text like:
```
include stdio stdlib typedef struct floatList  float list int size FloatList floatcmp const void if return else median fl qsort sizeof main static floats1 flist1 floats2 flist2 printf is 2f 85 60
median length 992021 000473 496010
```
order of words here does not matter.

Looking at original content, we see there is many non-alphanumeric characters which for human are helpful to determine language of a file.
We may try to create model which focus on that non-alphanumeric characters.

For that, we may 'reduce' original content to remove words and digits in order to create a text which more exposes non-alphanumeric characters.

Below is example of such reduction, applied to mentioned example content of file body.
```
#A <A.h>
#A <A.h>

A A A {
    A *A;
    A   A;
} *A;

A A( A A *a, A A *b) {
    A (*(A A *)a < *(A A *)b) A -D;
    A A *(A A *)a > *(A A *)b;
}

A A( A A )
{
    A( A->A, A->A, A(A), A);
    A D.D * ( A->A[A->A/D] + A->A[(A->A-D)/D]);
}

A A()
{
    A A A[] = { D.D, D.D, D.D, D.D, D.D, D.D };
    A A A A = { A, A(A)/A(A) };

    A A A[] = { D.D, D.D, D.D, D.D, D.D };
    A A A A = { A, A(A)/A(A) };

    A("A A A %D.A\n", A(&A)); /* D.A */
    A("A A A %D.A\n", A(&A)); /* D.A */
    A D;
}
```
Based on such text, we may try to crate a model which takes into account non-alphanumeric characters.

### Methodology of evaluation

Evaluation is done with 5-Fold stratified Crosvalidation. Files with same `proj_id` are never splited into train and validation set in order to have more meaningful results.

For evaluation of every model, there is done 5-Fold Crosvalidation. From each loop of Crosvalidation there is done:
 - train model on train set from Crosvalidation
 - evaluate metrics on validation set from Crosvalidation, used metrics:
       - `acc` - accuracy
       - `precision` - precision
       - `recall` - recall
       - `fscore` - fscore
After evaluation of each model, for each metrics is caluluated mean of specific metric value as well as stddev.
Mean and stddev give image how each metrics behaves in Crosvalidation and how much traing is stable.

For every set of evaluated model, there is created `csv` file with evaluated metrics. For every evaluation there are 2 files:
 - file with suffix `.detail.csv` contains metrics calucated for validation for every loop of Crosvalidation
 - file with suffix `.agg.csv` contains mean and stddev of metrics calucated over Crosvalidation loops

### Models evaluation

Evaluated models:
 - SVM fed by bag of words from input documents
 - SVM fed by bag of characters from input documents reduced by alphanumeric words
 - neural networks fed by bag of characters from input documents reduced by alphanumeric words

#### Run of models evaluation

Evaluation may be done, when input data file `data.csv` in located in repository root. Then to run evaluation of specific groups
of ML models it need to be run:
```
 $ python basic_models.py
 $ python basic_models_non_alphanumeric.py
 $ python pytorch_neural_models.py
```

`csv` files with results of evaluation were pushed to root of repository.

Each of these commands generates `csv` files with results of evaluation of model. 

### Technical things to done

In this exercise there was focus on ML part. Because of time limits some technical things need to be done:
 - create `requirements.txt` file in order to install all dependencies
 - more Python files to Python package
 - prepare unit tests
 - create `setup.py` file
 - create docstrings

## Conclusion
Work was started with investigating the input data. Found characteristic of data which are important with Crosvalidation process.
Created basic framework for automatically evaluating models based on Crosvalidation. Results from Crosvalidation loops are aggregated in order to
easier interpret data.

Evaluated basic ML model in order to have a baseline. All more advanced ML model are going to be compared with base line.

Evaluated few neural network models. All of them do not perform better than basic ML models which are baseline.
Result are in `csv` files which are pushed to repository.

Over evaluated models, the best was SVM model fed with words extracted from file body.
On validation set, achieved accuracy mean is about `0.70` with precision mean about`0.85` and recall mean about `0.68`.
F-score mean is about `0.73`.

### Future plans
 - evaluate TF-IDF vectorizer
 - try with different neural networks architectures
 - train word2vec based on given documents
 - evaluate SVM models with changed default parameters
 - evaluate other sklearn model which are quick to implement in current framework