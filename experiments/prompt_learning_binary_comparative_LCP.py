# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:41:02 2023

@author: knorth8
"""

import openai        
import csv
import pandas as pd
from tqdm import tqdm # nice loading bar.
from statistics import mode

openai.api_key = "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"

   

def promptLearning_LCP(word1, word2, sent1, sent2, gold_complexity):
    
    """ ===============================================================
        =================== Binaru Comparative LCP ====================
        ==============================================================="""
    
    "> English <"
    prompt1 =f"Which word is more difficult: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"   
        
    # prompt2 =f"Which word is more difficult to read: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
                              
    # prompt3 =f"Which word is more difficult to understand: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
    
    # prompt4 =f"Which word is learned first by a child: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
             
    prompt5 =f"Which word is more common: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
            
    # prompt6 =f"Which word is more familiar to the average person: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
    
    prompt7 =f"Which sentence is more difficult: (a). \"{sent1}\" or (b). \"{sent2}\"?\n"\
            f"Answer:"
        

               
   
    def OpenAi_promptLearning(prompt):
            # print(prompt)
            response = openai.Completion.create(
                model="text-davinci-003", # prior version is text-davinci-002 and text-davinci-003. Also, there is gpt-3.5-turbo
                prompt=prompt,
                stream=False,
                max_tokens=5,
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


    binary_answer1 = 1
    # binary_answer2 = 1
    # binary_answer3 = 1
    # binary_answer4 = 1
    binary_answer5 = 1
    # binary_answer6 = 1
    binary_answer7 = 1
    
    # try: # if we lose connection...
    prompt1_answer = OpenAi_promptLearning(prompt1)
    if word2.lower() in prompt1_answer.lower() and word1.lower() not in prompt1_answer.lower():
        binary_answer1 = 0
                
    # prompt2_answer = OpenAi_promptLearning(prompt2)
    # if word2 in prompt2_answer and word1 not in prompt2_answer:
    #     binary_answer2 = 0

    # prompt3_answer = OpenAi_promptLearning(prompt3)
    # if word2 in prompt3_answer and word1 not in prompt3_answer:
    #     binary_answer3 = 0

    # prompt4_answer = OpenAi_promptLearning(prompt4)
    # if word1 in prompt4_answer and word2 not in prompt4_answer:
    #     binary_answer4 = 0

    prompt5_answer = OpenAi_promptLearning(prompt5)
    if word1.lower() in prompt5_answer.lower() and word2.lower() not in prompt5_answer.lower():
        binary_answer5 = 0
    
    # prompt6_answer = OpenAi_promptLearning(prompt6)
    # if word1 in prompt6_answer and word2 not in prompt6_answer:
    #     binary_answer6 = 0
    
    prompt7_answer = OpenAi_promptLearning(prompt7)
    if "(b)." in prompt7_answer.lower() and "(a)." not in prompt7_answer.lower():
        binary_answer7 = 0
        
        # prompt8_answer = OpenAi_promptLearning(prompt8)


    # except:
        # norm_val1 = 0
        # norm_val2 = 0
        # norm_val3 = 0
        # norm_val4 = 0
        # norm_val5 = 0
        
            
    mode_complexity =  mode([binary_answer1, binary_answer5, binary_answer7])
    # prompt6_row = [sent, word, gold_complexity] + [avg_complexity, 0]
    
    # print(gold_complexity, norm_val1, norm_val2, norm_val3, norm_val4, norm_val5, avg_complexity)
    # input("Check answer")
    
    # prompt_row = [word1, word2, sent1, sent2, gold_complexity] + [prompt1_answer, prompt2_answer, prompt3_answer, prompt4_answer, prompt5_answer, prompt6_answer, prompt7_answer,
    #                                                               binary_answer1, binary_answer2, binary_answer3, binary_answer4, binary_answer5, binary_answer6, binary_answer7, mode_complexity]
        
    
        
    prompt_row = [word1, word2, sent1, sent2, gold_complexity] + [prompt1_answer, prompt5_answer, prompt7_answer, binary_answer1, binary_answer5, binary_answer7, mode_complexity]
        

    # print(prompt_row)
    # input("Enter")

        
    return prompt_row


def write_output(final_labels, prompt_name):
    
    """
        :description: This function writes the extracted data to .tsv files.
        :param data: extracted data to be written to .tsv.
        :return: .tsv files.
    """
    "---- Output ----"

    file_name = f"./prompt_learning_output/gpt_3_5_round_2_results/{prompt_name}.tsv"

    with open(file_name, 'w', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
        tsv_output = csv.writer(f_output, delimiter='\t')
        
        # headers_output = ["word1", "word2","sent1", "sent2","Gold_Complexity",
        #                   "prompt1_answer", "prompt2_answer", "prompt3_answer", "prompt4_answer", "prompt5_answer",
        #                   "prompt6_answer", "prompt7_answer", "binary_answer1", "binary_answer2", "binary_answer3", 
        #                   "binary_answer4", "binary_answer5", "binary_answer6", "binary_answer7", "mode_complexity"]
        
        
        headers_output = ["word1", "word2","sent1", "sent2","Gold_Complexity",
                          "prompt1_answer", "prompt5_answer", "prompt7_answer", 
                          "binary_answer1", "binary_answer5", "binary_answer7", "mode_complexity"]
        
        tsv_output.writerow(headers_output)
        
        for row in final_labels:
            tsv_output.writerow(row)
        
        print("\n\n==== Dataset Info ====")    
        print("Output saved as:", file_name)    



if __name__ == '__main__':
    
    # input_df = pd.read_csv("data_debug/v0.02_MultiLex_test.tsv", sep="\t") # used for debugging.
    final_dataset =[] 
    match_found = 0
    
    genre_list = ["all_genres", "bible", "biomed", "news"]
    
    for genre in genre_list:
    
        "--- Binary Datasets: Same as Tharindu Datasets --- "
        
        test_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\PT\{genre}\small_binary_comparative_MultiLex_test_PT.tsv"
        input_df = pd.read_csv(test_file, sep="\t") # CompLex EN test set.
        test = input_df[["word1", "word2", "context", "context2", "binary_comparative_val"]] # CompLex EN test set.
        

        with tqdm(total=len(test)) as pbar:
            for i in range(len(test)):
                
                # 'Old LCP columns'
            
                # genre = test["corpus"][i]
                # sent = test["sentence"][i]
                # word =  test["token"][i]
                
                'Binary Comparative LCP columns'
                
                word1 = test["word1"][i]
                word2 = test["word2"][i]
                sent1 = test["context"][i]
                sent2 = test["context2"][i]
                
                pbar.set_description(f"Predicting {genre}: ({word1} V {word2})")
                
                gold_complexity =  test["binary_comparative_val"][i]
                
                prompt_row = promptLearning_LCP(word1, word2, sent1, sent2, gold_complexity)
                final_dataset.append(prompt_row)
                pbar.update(1)
    
                # print(prompt_row)
        
        
            
        pbar.close()     
        write_output(final_dataset, f"v2_PT_{genre}_binary_comparative_LCP")
        final_dataset =[] 

        # input("Continue. Enter?")
