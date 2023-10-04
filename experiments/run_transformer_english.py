import os
import shutil

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from algo.transformer_model.evaluation import pearson_corr, spearman_corr, print_stat
from algo.transformer_model.model_args import LCPArgs
from algo.transformer_model.run_model import LCPModel

train = pd.read_csv("data/lcp_single_train.tsv", sep="\t")
test = pd.read_csv("data/lcp_single_test.tsv", sep="\t")

train = train[["corpus", "sentence", "token", "complexity"]]
test = test[["corpus", "sentence", "token", "complexity"]]

train["text_a"] = train["corpus"] + ' ' + train["token"]
test["text_a"] = test["corpus"] + ' ' + test["token"]

train = train.rename(columns={'sentence': 'text_b', 'complexity': 'labels'}).dropna()
test = test.rename(columns={'sentence': 'text_b', 'complexity': 'labels'}).dropna()

train = train[["text_a", "text_b", "labels"]]
test = test[["text_a", "text_b", "labels"]]

train, dev = train_test_split(train, test_size=0.2)

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
test_preds = np.zeros((len(test), 5))

portuguese_test = pd.read_csv("data/v0.02_MultiLex_test.tsv", sep="\t")
portuguese_test = portuguese_test[["genre", "pt_sentence", "pt_word", "avg_complexity"]]
portuguese_test["text_a"] = portuguese_test["genre"] + ' ' + portuguese_test["pt_word"]
portuguese_test = portuguese_test.rename(columns={'pt_sentence': 'text_b', 'avg_complexity': 'labels'}).dropna()

portuguese_test = portuguese_test[["text_a", "text_b", "labels"]]

portuguese_test_sentence_pairs = list(map(list, zip(portuguese_test['text_a'].to_list(), portuguese_test['text_b'].to_list())))
portuguese_test_preds = np.zeros((len(portuguese_test), 5))

for i in range(5):
    model_args = LCPArgs()
    model_args.best_model_dir = "english_outputs/mbert/best_model"
    model_args.eval_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 300
    model_args.evaluate_during_training_verbose = True
    model_args.logging_steps = 300
    model_args.learning_rate = 2e-5
    model_args.manual_seed = 777 * i
    model_args.max_seq_length = 256
    model_args.model_type = "bert"
    model_args.model_name = "bert-base-multilingual-cased"
    model_args.num_train_epochs = 5
    model_args.output_dir = "english_outputs/mbert/"
    model_args.overwrite_output_dir = True
    model_args.save_steps = 300
    model_args.train_batch_size = 8
    model_args.wandb_project = "LCP"
    model_args.regression = True

    if os.path.exists(model_args.output_dir) and os.path.isdir(
            model_args.output_dir):
        shutil.rmtree(model_args.output_dir)
    model = LCPModel(model_args.model_type, model_args.model_name, num_labels=1, use_cuda=torch.cuda.is_available(),
                     args=model_args)

    model.train_model(train, eval_df=dev, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                      mae=mean_absolute_error)
    model = LCPModel(model_args.model_type, model_args.best_model_dir, num_labels=1,
                     use_cuda=torch.cuda.is_available(), args=model_args)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    test_preds[:, i] = predictions
    portuguese_predictions, portuguese_raw_outputs = model.predict(portuguese_test_sentence_pairs)
    portuguese_test_preds[:, i] = portuguese_predictions

test['predictions'] = test_preds.mean(axis=1)
print_stat(test, 'labels', 'predictions')
test.to_csv("test_predictions.csv", sep="\t", index=False)


def genre_function(text):
    text = str(text)
    return text.split()[0]


test["genre"] = test['text_a'].apply(genre_function)

genres = test['genre'].unique()
for genre in genres:
    print(genre)
    filtered_predictions = test.loc[test['genre'] == genre]
    print_stat(filtered_predictions, 'labels', 'predictions')
    print("=================")

print("Portuguese predictions")

portuguese_test['predictions'] = portuguese_test_preds.mean(axis=1)
print_stat(portuguese_test, 'labels', 'predictions')
portuguese_test.to_csv("portuguese_test_predictions.csv", sep="\t", index=False)

portuguese_test["genre"] = portuguese_test['text_a'].apply(genre_function)

genres = portuguese_test['genre'].unique()
for genre in genres:
    print(genre)
    filtered_predictions = portuguese_test.loc[portuguese_test['genre'] == genre]
    print_stat(filtered_predictions, 'labels', 'predictions')
    print("=================")
