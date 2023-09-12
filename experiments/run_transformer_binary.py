import os
import shutil
import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from algo.transformer_model.evaluation import pearson_corr, spearman_corr, print_stat
from algo.transformer_model.model_args import LCPArgs
from algo.transformer_model.run_model import LCPModel

train = pd.read_csv("data/binary_comparative_MultiLex_train_PT.tsv", sep="\t")
dev = pd.read_csv("data/binary_comparative_MultiLex_dev_PT.tsv", sep="\t")
test = pd.read_csv("data/binary_comparative_MultiLex_test_PT.tsv", sep="\t")


def modify_sentence(word, context):
    new_word = "<" + word + ">"
    return context.replace(word, new_word)


train['context1'] = train.apply(lambda row: modify_sentence(row['word1'], row['context']), axis=1)
train['context2'] = train.apply(lambda row: modify_sentence(row['word2'], row['context2']), axis=1)

dev['context1'] = dev.apply(lambda row: modify_sentence(row['word1'], row['context']), axis=1)
dev['context2'] = dev.apply(lambda row: modify_sentence(row['word2'], row['context2']), axis=1)

test['context1'] = test.apply(lambda row: modify_sentence(row['word1'], row['context']), axis=1)
test['context2'] = test.apply(lambda row: modify_sentence(row['word2'], row['context2']), axis=1)

train = train[["context1", "context2", "binary_comparative_val"]]
dev = dev[["context1", "context2", "binary_comparative_val"]]
test = test[["context1", "context2", "binary_comparative_val"]]

train = train.rename(columns={'context1': 'text_a', 'context2': 'text_b', 'binary_comparative_val': 'labels'}).dropna()
dev = dev.rename(columns={'context1': 'text_a', 'context2': 'text_b', 'binary_comparative_val': 'labels'}).dropna()
test = test.rename(columns={'context1': 'text_a', 'context2': 'text_b', 'binary_comparative_val': 'labels'}).dropna()

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
test_preds = np.zeros((len(test), 5))

for i in range(5):
    model_args = LCPArgs()
    model_args.best_model_dir = "portuguese_outputs/best_model"
    model_args.eval_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 120
    model_args.evaluate_during_training_verbose = True
    model_args.logging_steps = 120
    model_args.learning_rate = 2e-5
    model_args.manual_seed = 777 * i
    model_args.max_seq_length = 256
    model_args.model_type = "xlmroberta"
    model_args.model_name = "xlm-roberta-large"
    model_args.num_train_epochs = 5
    model_args.output_dir = "portuguese_outputs/"
    model_args.save_steps = 120
    model_args.train_batch_size = 8
    model_args.wandb_project = "LCP"
    model_args.regression = True

