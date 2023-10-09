# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:42:03 2023

@author: knorth8
"""

import pandas as pd


gpt_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\prompt_learning_output\TheBloke-Llama-2-7b-chat-fp16_EN_all_genres_binary_comparative_LCP.tsv"

test_small_file = r"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\EN\all_genres\small_binary_comparative_MultiLex_test_EN.tsv"

gpt_df = pd.read_csv(gpt_file, sep="\t") # CompLex EN test set.
test_file_df = pd.read_csv(test_small_file, sep="\t") # CompLex EN test set.


correct_ids = [id_num for id_num in test_file_df["id"]] # extracts correct ids from df. 

new_df = gpt_df[gpt_df['id'].isin(correct_ids)] # gets only those rows which match values in list: correct_ids. 



df_a = pd.DataFrame({'VAL2' : correct_ids})

new_df = pd.merge(df_a, new_df,left_on='VAL2',right_on='id',how='outer')
new_df = new_df.drop('VAL2', axis=1)

new_df.to_csv(r"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\prompt_learning_output\TheBloke-Llama-2-7b-chat-fp16_small_EN_all_genres_binary_comparative_LCP.tsv", "\t", encoding="utf-8", index=False)

print(new_df)