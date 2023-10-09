# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:40:18 2023

@author: knorth8
"""

import pandas as pd
from tqdm import tqdm # nice loading bar.

df_genres = pd.read_csv("combined_binary_comparative_MultiLex_test_EN.tsv", sep='\t', encoding ="utf-8")
df_prompt_results = pd.read_csv("EN_all_genres_binary_comparative_LCP.tsv", sep='\t', encoding ="utf-8")

genre_match_list = []
unknowns = []
correct_match = []

with tqdm(total=len(df_prompt_results)) as pbar:
    
    for i in range(len(df_prompt_results)):
        pbar.set_description(f"Found: {len(correct_match)}/{len(df_prompt_results)}, Not Found: {len(unknowns)}/{len(df_prompt_results)}")
        prompt_word1 = df_prompt_results["word1"][i]
        prompt_word2 = df_prompt_results["word2"][i]
        prompt_sent1 = df_prompt_results["sent1"][i]
        prompt_sent2 = df_prompt_results["sent2"][i]
        count=0
        
        for x in range(len(df_genres)):
            genre_word1 = df_genres["word1"][x]
            genre_word2 = df_genres["word2"][x]
            genre_sent1 = df_genres["context"][x]
            genre_sent2 = df_genres["context2"][x]
            
            
            if prompt_word1 == genre_word1 and prompt_word2 == genre_word2 and prompt_sent1 == genre_sent1 and prompt_sent2 == genre_sent2:
                genre =  df_genres["genre"][x]
                # print(f"Match Genre: {genre}")
                count+=1
                genre_match_list.append(genre)
                correct_match.append(genre)
                # input("enter")
            
            
        if count == 0:
            genre_match_list.append("unknown")
            unknowns.append("unknown")
            
            
                
        pbar.update(1)
    
pbar.close() 
df_prompt_results["genre"] = genre_match_list

print("Match found: {len(genre_match_list)}/{len(df_prompt_results)}")

df_prompt_results.to_csv('EN_genre_split_binary_comparative_LCP.tsv.tsv', sep="\t")