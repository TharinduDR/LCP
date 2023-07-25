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

for i in range(5):
    model_args = LCPArgs()
    model_args.eval_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 120
    model_args.evaluate_during_training_verbose = True
    model_args.logging_steps = 120
    model_args.learning_rate = 5e-6
    model_args.manual_seed = 777*i
    model_args.max_seq_length = 256
    model_args.model_type = "bert"
    model_args.model_name = "bert-large-cased"
    model_args.num_train_epochs = 5
    model_args.save_steps = 120
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

test['predictions'] = test_preds.mean(axis=1)
print_stat(test, 'labels', 'predictions')
test.to_csv("test_predictions.csv", sep="\t", index=False)
