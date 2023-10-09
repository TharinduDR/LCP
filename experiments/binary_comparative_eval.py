# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:18:05 2023

@author: knorth8
"""

import pandas as pd
from algo.transformer_model.evaluation import pearson_corr, spearman_corr, print_binary_stat, return_binary_stat

# prompts = ["binary_answer1", "binary_answer2", "binary_answer3", 
#             "binary_answer4", "binary_answer5", "binary_answer6", 
#             "binary_answer7", "mode_complexity"]



# prompts = ["binary_answer1", "binary_answer5",
#             "binary_answer7", "mode_complexity"] # GPT 3.5 round 2

prompts = ["binary_answer1", "binary_answer2", "binary_answer3", 
            "mode_complexity"] # Falcon list.

# prompts = ["binary_value1", "binary_value2", "binary_value3", 
#             "binary_value4"] # Llama 2 list.

# prompts = ["binary_value1", "binary_value2", "binary_value3", 
#             "binary_value4"] # Llama 2 list.



genre_list = ["all_genres", "bible", "biomed", "news"]
# genre_list = ["biomed"]


for prompt_type in prompts:
    prior_val =f"& & mosaicml-mpt-7b-instruct-{prompt_type.replace('binary_answer', 'prompt')}"
    
    for genre in genre_list:
        input_df = pd.read_csv(f"prompt_learning_output/MPT-7B_round2_results/mosaicml-mpt-7b-instruct_small_EN_{genre}_binary_comparative_LCP.tsv", sep="\t")
        # input_df = pd.read_csv(f"prompt_learning_output/gpt_3_5_round_2_results/v2_PT_{genre}_binary_comparative_LCP.tsv", sep="\t")
        # input_df = pd.read_csv(f"prompt_learning_output/Llama_round_2_results/TheBloke-Llama-2-7b-chat-fp16_small_EN_{genre}_binary_comparative_LCP.tsv", sep="\t")

        stats = return_binary_stat(input_df, 'Gold_Complexity', prompt_type)
        # stats = print_binary_stat(input_df, 'binary_comparative_val', prompt_type)

        for val in stats:
            rounded_val = round(val, 4)
            if len(str(rounded_val)) == 5:
                rounded_val = str(rounded_val)+ "0"
                
          
            prior_val = prior_val + " & " + str(rounded_val)
       
            
    prior_val = prior_val + "\\\\" 
    print(prior_val)
        
            
        # input("enter")
    