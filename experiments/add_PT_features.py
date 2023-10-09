# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:58:32 2023

@author: knorth8
"""

import pandas as pd
from wordfreq import word_frequency


lang_list = ["EN", "PT"]
genre_list = ["all_genres", "bible", "biomed", "news"]

for language in lang_list:

    for genre in genre_list:
        train_file =fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data\binary_comparative_MultiLex\EN\{genre}\binary_comparative_MultiLex_train_EN.tsv"
        # test_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\{language}\{genre}\small_binary_comparative_MultiLex_test_{language}.tsv"
        input_df = pd.read_csv(train_file, sep="\t") # CompLex EN test set.
        
        
        input_df["word1s_PT_freqs"] = [word_frequency(word, 'pt') for word in input_df["word1"]] # adds PT word freqs to df.
        input_df["word2s_PT_freqs"]  = [word_frequency(word, 'pt') for word in input_df["word2"]]
        
        # print(word1s_freqs)
        
        # print(input_df)
        # input("enter")
        
        input_df.to_csv(fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data\binary_comparative_MultiLex\{language}\{genre}\newfeatures_og_binary_comparative_MultiLex_train{language}.tsv", "\t", encoding="utf-8", index=False)
        print(fr"\nC:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data\binary_comparative_MultiLex\{language}\{genre}\newfeatures_og_binary_comparative_MultiLex_train{language}.tsv")