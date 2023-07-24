import os
import shutil

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from transformer_model.evaluation import pearson_corr, spearman_corr, print_stat
from transformer_model.model_args import LCPArgs
from transformer_model.run_model import LCPModel

train = pd.read_csv("data/v0.01_CompLex-pt_train.tsv", sep="\t")
dev = pd.read_csv("data/v0.01_CompLex-pt_dev.tsv", sep="\t")
test = pd.read_csv("data/v0.01_CompLex-pt_test.tsv", sep="\t")

train = train[["genre", "pt_sentence", "pt_word", "avg_complexity"]]
dev = dev[["genre", "pt_sentence", "pt_word", "avg_complexity"]]
test = test[["genre", "pt_sentence", "pt_word", "avg_complexity"]]

train["text_a"] = train["genre"] + ' ' + train["pt_word"]
dev["text_a"] = dev["genre"] + ' ' + dev["pt_word"]
test["text_a"] = test["genre"] + ' ' + test["pt_word"]

train = train.rename(columns={'pt_sentence': 'text_b', 'avg_complexity': 'labels'}).dropna()
dev = dev.rename(columns={'pt_sentence': 'text_b', 'avg_complexity': 'labels'}).dropna()
test = test.rename(columns={'pt_sentence': 'text_b', 'avg_complexity': 'labels'}).dropna()

train = train[["text_a", "text_b", "labels"]]
dev = dev[["text_a", "text_b", "labels"]]
test = test[["text_a", "text_b", "labels"]]

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
test_preds = np.zeros((len(test), 5))
for i in range(5):

    args = LCPArgs()
    args.eval_batch_size = 16
    args.evaluate_during_training = True
    args.evaluate_during_training_steps = 120
    args.evaluate_during_training_verbose = True
    args.logging_steps = 120
    args.learning_rate = 1e-5
    args.manual_seed = 777*i
    args.max_seq_length = 120
    args.model_type = "bert"
    args.model_name = "bert-base-multilingual-cased"
    args.num_train_epochs = 5
    args.save_steps = 120
    args.train_batch_size = 8

    if os.path.exists(args.output_dir) and os.path.isdir(
            args.output_dir):
        shutil.rmtree(args.output_dir)
        model = LCPModel(args.model_type, args.model_name, num_labels=1, use_cuda=torch.cuda.is_available(),
                                    args=args)
        model.train_model(train, eval_df=dev, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = LCPModel(args.model_type, args.best_model_dir, num_labels=1,
                                    use_cuda=torch.cuda.is_available(), rgs=args)
        predictions, raw_outputs = model.predict(test_sentence_pairs)
        test_preds[:, i] = predictions

test['predictions'] = test_preds.mean(axis=1)
print_stat(test, 'labels', 'predictions')
