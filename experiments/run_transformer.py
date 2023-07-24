import pandas as pd


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