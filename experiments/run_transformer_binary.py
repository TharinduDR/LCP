import os
import shutil
import math
import pandas as pd
import numpy as np
import sklearn
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from algo.transformer_model.evaluation import pearson_corr, spearman_corr, print_stat, print_binary_stat, macro_f1, \
    weighted_f1
from algo.transformer_model.model_args import LCPArgs
from algo.transformer_model.run_model import LCPModel



train_bible = pd.read_csv("data/binary_comparative_MultiLex/PT/bible/binary_comparative_MultiLex_train_PT.tsv", sep="\t")
train_biomed = pd.read_csv("data/binary_comparative_MultiLex/PT/biomed/binary_comparative_MultiLex_train_PT.tsv", sep="\t")
train_news = pd.read_csv("data/binary_comparative_MultiLex/PT/news/binary_comparative_MultiLex_train_PT.tsv", sep="\t")

dev_bible = pd.read_csv("data/binary_comparative_MultiLex/PT/bible/binary_comparative_MultiLex_dev_PT.tsv", sep="\t")
dev_biomed = pd.read_csv("data/binary_comparative_MultiLex/PT/biomed/binary_comparative_MultiLex_dev_PT.tsv", sep="\t")
dev_news = pd.read_csv("data/binary_comparative_MultiLex/PT/news/binary_comparative_MultiLex_dev_PT.tsv", sep="\t")

test_bible = pd.read_csv("data/binary_comparative_MultiLex/PT/bible/binary_comparative_MultiLex_test_PT.tsv", sep="\t")
test_biomed = pd.read_csv("data/binary_comparative_MultiLex/PT/biomed/binary_comparative_MultiLex_test_PT.tsv", sep="\t")
test_news = pd.read_csv("data/binary_comparative_MultiLex/PT/news/binary_comparative_MultiLex_test_PT.tsv", sep="\t")

train_bible["genre"] = "bible"
train_biomed["genre"] = "biomed"
train_news["genre"] = "news"

dev_bible["genre"] = "bible"
dev_biomed["genre"] = "biomed"
dev_news["genre"] = "news"

test_bible["genre"] = "bible"
test_biomed["genre"] = "biomed"
test_news["genre"] = "news"

train = pd.concat([train_bible, train_biomed, train_news])
dev = pd.concat([dev_bible, dev_biomed, dev_news])
test = pd.concat([test_bible, test_biomed, test_news])

def modify_sentence(word, context):
    new_word = "<" + word + ">"
    return context.replace(word, new_word)


train['context1'] = train.apply(lambda row: modify_sentence(row['word1'], row['context']), axis=1)
train['context2'] = train.apply(lambda row: modify_sentence(row['word2'], row['context2']), axis=1)

dev['context1'] = dev.apply(lambda row: modify_sentence(row['word1'], row['context']), axis=1)
dev['context2'] = dev.apply(lambda row: modify_sentence(row['word2'], row['context2']), axis=1)

test['context1'] = test.apply(lambda row: modify_sentence(row['word1'], row['context']), axis=1)
test['context2'] = test.apply(lambda row: modify_sentence(row['word2'], row['context2']), axis=1)

train = train.dropna(subset=['binary_comparative_val'])  # Remove rows with NaN or infinity
train['binary_comparative_val'] = train['binary_comparative_val'].astype(int)

dev = dev.dropna(subset=['binary_comparative_val'])  # Remove rows with NaN or infinity
dev['binary_comparative_val'] = dev['binary_comparative_val'].astype(int)

test = test.dropna(subset=['binary_comparative_val'])  # Remove rows with NaN or infinity
test['binary_comparative_val'] = test['binary_comparative_val'].astype(int)

train = train[["genre", "context1", "context2", "binary_comparative_val"]]
dev = dev[["genre", "context1", "context2", "binary_comparative_val"]]
test = test[["genre", "context1", "context2", "binary_comparative_val"]]

train = train.rename(columns={'context1': 'text_a', 'context2': 'text_b', 'binary_comparative_val': 'labels'}).dropna()
dev = dev.rename(columns={'context1': 'text_a', 'context2': 'text_b', 'binary_comparative_val': 'labels'}).dropna()
test = test.rename(columns={'context1': 'text_a', 'context2': 'text_b', 'binary_comparative_val': 'labels'}).dropna()

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
test_preds = np.zeros((len(test), 5))

for i in range(5):
    model_args = LCPArgs()
    model_args.best_model_dir = "portuguese_binary_outputs/best_model"
    model_args.eval_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1300
    model_args.evaluate_during_training_verbose = True
    model_args.logging_steps = 1300
    model_args.learning_rate = 2e-5
    model_args.manual_seed = 777 * i
    model_args.max_seq_length = 256
    model_args.model_type = "xlmroberta"
    model_args.model_name = "xlm-roberta-large"
    model_args.num_train_epochs = 5
    model_args.output_dir = "portuguese_binary_outputs/"
    model_args.save_steps = 1300
    model_args.train_batch_size = 8
    model_args.wandb_project = "LCP_BINARY"
    model_args.regression = False

    if os.path.exists(model_args.output_dir) and os.path.isdir(
            model_args.output_dir):
        shutil.rmtree(model_args.output_dir)
    model = LCPModel(model_args.model_type, model_args.model_name, use_cuda=torch.cuda.is_available(),
                     args=model_args)
    model.train_model(train, eval_df=dev, macro_f1=macro_f1, weighted_f1=weighted_f1)
    model = LCPModel(model_args.model_type, model_args.best_model_dir,
                     use_cuda=torch.cuda.is_available(), args=model_args)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    test_preds[:, i] = predictions

final_predictions = []
for pred_row in test_preds:
    all_predictions = pred_row.tolist()
    final_predictions.append(int(max(set(all_predictions), key=all_predictions.count)))

test['predictions'] = final_predictions

print_binary_stat(test, 'labels', 'predictions')
test.to_csv("test_predictions.csv", sep="\t", index=False)

genres = test['genre'].unique()
for genre in genres:
    print(genre)
    filtered_predictions = test.loc[test['genre'] == genre]
    print_binary_stat(filtered_predictions, 'labels', 'predictions')
    print("=================")


