# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:01:19 2023

@author: knorth8
"""


import pandas as pd
from algo.transformer_model.evaluation import pearson_corr, spearman_corr, print_stat_v2

input_df = pd.read_csv("prompt_learning_output/Llama_LCP_results/TheBloke-Llama-2-7b-chat-fp16_PT_LCP.tsv", sep="\t")

prompts = ["prompt1_pred","prompt2_pred","prompt3_pred","prompt4_pred","avg_pred1_2","avg_pred3_4"]
genres = ["bible", "news", "biomed"]
# genres = ["bible", "europarl", "biomed"]





for prompt_type in prompts:

    scores = f"& 0 &Llama2-{prompt_type.replace('_pred', '').replace('_', '-')}"
    prior_val =""
    
    stats = print_stat_v2(input_df, 'Gold_Complexity', prompt_type)

    for val in stats:
        rounded_val = round(val, 4)
        if len(str(rounded_val)) == 5:
            rounded_val = str(rounded_val)+ "0"
            
      
        prior_val = prior_val + " & " + str(rounded_val)

    for genre in genres:
    
        genre_df = input_df.loc[input_df['Genre'] == genre]
        
        # print(genre)
    
        stats = print_stat_v2(genre_df, 'Gold_Complexity', prompt_type)
    
        for val in stats:
            rounded_val = round(val, 4)
            if len(str(rounded_val)) == 5:
                rounded_val = str(rounded_val)+ "0"
                
          
            prior_val = prior_val + " & " + str(rounded_val)
            # print(prior_val)
            
    
        
    score_row = scores+prior_val +" \\\\"
    
    print(score_row)
    # input("enter")

