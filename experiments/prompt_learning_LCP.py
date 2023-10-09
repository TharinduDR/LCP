# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:41:02 2023

@author: knorth8
"""

import openai        
import csv
import pandas as pd
from tqdm import tqdm # nice loading bar.

openai.api_key = "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"

   

def promptLearning_LCP(genre, sent, word, gold_complexity):
    
    """ ============================================
        =================== LCP ====================
        ============================================"""
    
    "> English <"
    prompt1 =f"On a scale for 1 to 5 with 5 being the most difficult, how difficult is the \"{word}\"?\n"\
            f"Answer:"   
        
    prompt2 =f"On a scale for 1 to 10 with 10 being the most difficult, how difficult is the \"{word}\"?\n"\
            f"Answer:"
                              
    prompt3 =f"Sentence: {sent}\n" \
             f"On a scale for 1 to 5 with 5 being the most difficult, how difficult is the \"{word}\" in the above sentence?\n"\
             f"Answer:"
    
    prompt4 =f"Sentence: {sent}\n" \
             f"On a scale for 1 to 10 with 10 being the most difficult, how difficult is the \"{word}\" in the above sentence?\n"\
             f"Answer:"
             
    prompt5 =f"On a scale for 1 to 10 with 10 being the most difficult, how difficult is the \"{word}\" for a child?\n"\
             f"Answer:"
               
   
    def OpenAi_promptLearning(prompt):
            # print(prompt)
            response = openai.Completion.create(
                model="text-davinci-003", # prior version is text-davinci-002 and text-davinci-003. Also, there is gpt-3.5-turbo
                prompt=prompt,
                stream=False,
                max_tokens=256,
                top_p=1,
                best_of=1,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            
            answer = response["choices"][0]["text"].lower().lstrip()
            
            return answer
        
    def normalize_val(val_string, min_val, max_val):
        if val_string.isnumeric() == True:
            # (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            norm_val = (int(val_string) - min_val)/(max_val-min_val)
            # print(f"{prompt1_answer} = {norm_val}")
        
        else:
            # print("Warning: Answer not Int")
            norm_val = 0.0 # returns 0.0 if answer provided by gpt 3.5 is not a number.
    
        return norm_val

    try: # if we lose connection...
        prompt1_answer = OpenAi_promptLearning(prompt1)
        norm_val1 = normalize_val(prompt1_answer, 1, 5)
        # prompt1_row = [sent, word, gold_complexity] + [norm_val1, prompt1_answer]
        
        prompt2_answer = OpenAi_promptLearning(prompt2)
        norm_val2 = normalize_val(prompt2_answer, 1, 10)
        # prompt2_row = [sent, word, gold_complexity] + [norm_val2, prompt2_answer]
        
        prompt3_answer = OpenAi_promptLearning(prompt3)
        norm_val3 = normalize_val(prompt3_answer, 1, 5)
        # prompt3_row = [sent, word, gold_complexity] + [norm_val3, prompt3_answer]
        
        prompt4_answer = OpenAi_promptLearning(prompt4)
        norm_val4 = normalize_val(prompt4_answer, 1, 10)
        # prompt4_row = [sent, word, gold_complexity] + [norm_val4, prompt4_answer]
        
        prompt5_answer = OpenAi_promptLearning(prompt5)
        norm_val5 = normalize_val(prompt5_answer, 1, 10)
        # prompt5_row = [sent, word, gold_complexity] + [norm_val5, prompt5_answer]

        
    except:
        norm_val1 = 0
        norm_val2 = 0
        norm_val3 = 0
        norm_val4 = 0
        norm_val5 = 0
        
            
    avg_complexity = sum([norm_val1, norm_val2, norm_val3, norm_val4, norm_val5])/5
    # prompt6_row = [sent, word, gold_complexity] + [avg_complexity, 0]
    
    # print(gold_complexity, norm_val1, norm_val2, norm_val3, norm_val4, norm_val5, avg_complexity)
    # input("Check answer")
    
    prompt_row = [genre, sent, word, gold_complexity] + [norm_val1, norm_val2, norm_val3, norm_val4, norm_val5, round(avg_complexity, 2)]
        

            #     

    # pbar.close()            
    return prompt_row


def write_output(final_labels, prompt_name):
    
    """
        :description: This function writes the extracted data to .tsv files.
        :param data: extracted data to be written to .tsv.
        :return: .tsv files.
    """
    "---- Output ----"

    file_name = f"./prompt_learning_output/{prompt_name}.tsv"

    with open(file_name, 'w', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
        tsv_output = csv.writer(f_output, delimiter='\t')
        
        headers_output = ["Genre", "Sentence","ComplexWord","Gold_Complexity","prompt1_pred","prompt2_pred","prompt3_pred","prompt4_pred","prompt5_pred","avg_pred"]
        tsv_output.writerow(headers_output)
        
        for row in final_labels:
            tsv_output.writerow(row)
        
        print("\n\n==== Dataset Info ====")    
        print("Output saved as:", file_name)    



if __name__ == '__main__':
    
    # input_df = pd.read_csv("data_debug/v0.02_MultiLex_test.tsv", sep="\t") # used for debugging.
    
    # input_df = pd.read_csv("data/v0.02_MultiLex_test.tsv", sep="\t") # MultiLex-PT PT-BR test set.
    # test = input_df[["genre", "pt_sentence", "pt_word", "avg_complexity"]] # MultiLex-PT PT-BR test set.
    
    input_df = pd.read_csv("data/lcp_single_test.tsv", sep="\t") # CompLex EN test set.
    test = input_df[["corpus", "sentence", "token", "complexity"]] # CompLex EN test set.
  
    
    final_dataset =[] 
    match_found = 0
    
    with tqdm(total=len(test)) as pbar:
        for i in range(len(test)):
            pbar.set_description(f"Predicting Complexity: (Instance #{i})")
            # 'MulitLex: PT-BR'
            # genre = test["genre"][i]
            # sent = test["pt_sentence"][i]
            # word =  test["pt_word"][i]
            # gold_complexity =  test["avg_complexity"][i]
            
            'CompLex: EN'
            genre = test["corpus"][i]
            sent = test["sentence"][i]
            word =  test["token"][i]
            gold_complexity =  round(test["complexity"][i], 2)

            
            prompt_row = promptLearning_LCP(genre, sent, word, gold_complexity)
            final_dataset.append(prompt_row)
            pbar.update(1)

            # print(prompt_row)
    
            # input("Continue. Enter?")
            
    pbar.close()     
    write_output(final_dataset, "EN_prompt_answers_LCP")

    
