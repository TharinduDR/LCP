# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:09:31 2023

@author: knorth8
"""
import pandas as pd
from sklearn.utils import shuffle


lang_list = ["EN", "PT"]
genre_list = ["all_genres", "bible", "biomed", "news"]

for language in lang_list:

    for genre in genre_list:
        test_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\{language}\{genre}\binary_comparative_MultiLex_test_{language}.tsv"
        input_df = pd.read_csv(test_file, sep="\t") # CompLex EN test set.
        
        new_df = shuffle(input_df) # randomizes df.
        
        percentage_10_dataset = len(new_df)*0.1
        
        new_df = new_df.head(int(percentage_10_dataset))

        new_df.to_csv(fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\{language}\{genre}\small_binary_comparative_MultiLex_test_{language}.tsv", "\t", encoding="utf-8", index=False)
        print(fr"\nC:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\{language}\{genre}\small_binary_comparative_MultiLex_test_{language}.tsv")