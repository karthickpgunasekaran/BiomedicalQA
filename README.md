# BiomedicalQA
This repository contains the code and preprocess datasets,for our participating model in BioASQ 8b challenge. We use BioSentVec, Scibert and Biobert language modelling representations for this task. The codebase was built on top of https://github.com/dmis-lab/bioasq-biobert





# Usage

## 1. Setup Python paths

```
$ export PYTHONPATH=<repo_root_dir>
```

## 2. Install requirements

```
$ pip install -r requirements.txt
```

## 3. Train a model

Use the following comand to see the posible models and training configurations.

```
$ python biobert/training.py --help

usage: training.py 

To level config for the BioBert project

optional arguments:


```

For instance, to train, execute the following:

```
$ python biobert/training.py 
```

## 4. Extract predictions from the test set using a trained model

```
python biobert/test.py 
```
Supply, `--resume_snapshot` and `experiment_dir` appropriately from your training run. (Look at the console log of your training to find the paths to these two).

# Running baselines

Please see the README file in the Baseline folder.


## Running unit tests (NOT MANDATORY)

```
$ chmod +x run_tests.sh
$ ./run_tests.sh
```

# Contributors
1. [Karthick Gunasekaran](https://github.com/karthickpgunasekaran)

2. [Vaishnavi](https://github.com/)

3. [Kun Li](https://github.com/)

