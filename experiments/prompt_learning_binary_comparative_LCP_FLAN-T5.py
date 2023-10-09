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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, FalconModel, FalconForCausalLM

openai.api_key = "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"

   

def promptLearning_LCP(id_num, word1, word2, sent1, sent2, gold_complexity, model, tokenizer):
    
    """ ===============================================================
        =================== Binary Comparative LCP ====================
        ==============================================================="""
    
    "> English <"
    prompt1 =f"Which word is more difficult: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"   
        
    prompt2 =f"Which word is more difficult to read: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
                              
    prompt3 =f"Which word is more difficult to understand: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
    
    prompt4 =f"Which word is learned first by a child: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
             
    prompt5 =f"Which word is more common: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
            
    prompt6 =f"Which word is more familiar to the average person: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
    
    prompt7 =f"Which sentence is more difficult: (a). \"{sent1}\" or (b). \"{sent2}\"?\n"\
            f"Answer:"
        

        
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
    binary_answer2 = 1
    binary_answer3 = 1
    binary_answer4 = 1
    binary_answer5 = 1
    binary_answer6 = 1
    binary_answer7 = 1
    
    
    " === for Downloaded Model ==="
    inputs = tokenizer(f"{prompt1}",return_tensors="pt")
    prompt1_answer = model.generate(**inputs, max_new_tokens=5)
    prompt1_answer = tokenizer.batch_decode(prompt1_answer, skip_special_tokens=True)[0]
    if word2.lower()  in prompt1_answer.lower()  and word1.lower()  not in prompt1_answer.lower() :
        binary_answer1 = 0
    
    inputs = tokenizer(f"{prompt2}",return_tensors="pt")
    prompt2_answer = model.generate(**inputs, max_new_tokens=5)
    prompt2_answer = tokenizer.batch_decode(prompt2_answer, skip_special_tokens=True)[0]
    if word2.lower()  in prompt2_answer.lower()  and word1.lower()  not in prompt2_answer.lower() :
        binary_answer2 = 0
        
    
    inputs = tokenizer(f"{prompt3}",return_tensors="pt")
    prompt3_answer = model.generate(**inputs, max_new_tokens=5)
    prompt3_answer = tokenizer.batch_decode(prompt3_answer, skip_special_tokens=True)[0]
    if word2.lower()  in prompt3_answer.lower()  and word1.lower()  not in prompt3_answer.lower() :
        binary_answer3 = 0
    

    
    inputs = tokenizer(f"{prompt4}",return_tensors="pt")
    prompt4_answer = model.generate(**inputs, max_new_tokens=5)
    prompt4_answer = tokenizer.batch_decode(prompt4_answer, skip_special_tokens=True)[0]
    if word1.lower()  in prompt4_answer.lower()  and word2.lower()  not in prompt4_answer.lower() :
        binary_answer4 = 0

    inputs = tokenizer(f"{prompt5}",return_tensors="pt")
    prompt5_answer = model.generate(**inputs, max_new_tokens=5)
    prompt5_answer = tokenizer.batch_decode(prompt5_answer, skip_special_tokens=True)[0]
    if word1.lower()  in prompt5_answer.lower()  and word2.lower()  not in prompt5_answer.lower() :
        binary_answer5 = 0
        
        
    inputs = tokenizer(f"{prompt6}",return_tensors="pt")
    prompt6_answer = model.generate(**inputs, max_new_tokens=5)
    prompt6_answer = tokenizer.batch_decode(prompt6_answer, skip_special_tokens=True)[0]
    if word1.lower()  in prompt6_answer.lower()  and word2.lower()  not in prompt6_answer.lower() :
        binary_answer6 = 0
    
    
    inputs = tokenizer(f"{prompt7}",return_tensors="pt")
    prompt7_answer = model.generate(**inputs, max_new_tokens=5)
    prompt7_answer = tokenizer.batch_decode(prompt7_answer, skip_special_tokens=True)[0]
    if "(b)." in prompt7_answer.lower()  and "(a)." not in prompt7_answer.lower() :
        binary_answer7 = 0

            
    mode_complexity =  mode([binary_answer1, binary_answer5,  binary_answer7])
    # prompt6_row = [sent, word, gold_complexity] + [avg_complexity, 0]
    
    
    prompt_row = [id_num, word1, word2, sent1, sent2, gold_complexity] + [prompt1_answer, prompt2_answer, prompt3_answer, prompt4_answer, prompt5_answer, prompt6_answer, prompt7_answer,
                                                                  binary_answer1, binary_answer2, binary_answer3, binary_answer4, binary_answer5, binary_answer6, binary_answer7, mode_complexity]

    print(prompt_row)
    input("Enter")

    # pbar.close()            
    return prompt_row


def write_output(final_labels, prompt_name, model_card):
    
    """
        :description: This function writes the extracted data to .tsv files.
        :param data: extracted data to be written to .tsv.
        :return: .tsv files.
    """
    "---- Output ----"

    file_name = f"./prompt_learning_output/flan-T5_round2_results/{model_card}_{prompt_name}.tsv"

    with open(file_name, 'w', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
        tsv_output = csv.writer(f_output, delimiter='\t')
        
        headers_output = ["id","word1", "word2","sent1", "sent2","Gold_Complexity",
                          "prompt1_answer", "prompt2_answer", "prompt3_answer", "prompt4_answer", "prompt5_answer",
                          "prompt6_answer", "prompt7_answer", "binary_answer1", "binary_answer2", "binary_answer3", 
                          "binary_answer4", "binary_answer5", "binary_answer6", "binary_answer7", "mode_complexity"]
        
        tsv_output.writerow(headers_output)
        
        for row in final_labels:
            tsv_output.writerow(row)
        
        print("\n\n==== Dataset Info ====")    
        print("Output saved as:", file_name)    



if __name__ == '__main__':
    
    # input_df = pd.read_csv("data_debug/v0.02_MultiLex_test.tsv", sep="\t") # used for debugging.
    final_dataset =[] 
    match_found = 0
    
    # model_card = "google/flan-t5-base"
    model_card = "Rocketknight1/falcon-rw-1b"
    # model_card = "meta-llama/Llama-2-7b-hf"
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_card)
    model = FalconForCausalLM.from_pretrained(model_card)
    
    # model = LlamaForCausalLM.from_pretrained(model_card)
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    
    
    model_card = model_card.replace("/","-")
    
    # genre_list = [ "bible"]
    genre_list = ["all_genres", "bible", "biomed", "news"]
    
    for genre in genre_list:
    
        "--- Binary Datasets: Same as Tharindu Datasets --- "
        
        test_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\EN\{genre}\small_binary_comparative_MultiLex_test_EN.tsv"
        input_df = pd.read_csv(test_file, sep="\t") # CompLex EN test set.
        test = input_df[["id","word1", "word2", "context", "context2", "binary_comparative_val"]] # CompLex EN test set.
        

        with tqdm(total=len(test)) as pbar:
            for i in range(len(test)):
                
                # 'Old LCP columns'
            
                # genre = test["corpus"][i]
                # sent = test["sentence"][i]
                # word =  test["token"][i]
                
                'Binary Comparative LCP columns'
                id_num = test["id"][i]
                word1 = test["word1"][i]
                word2 = test["word2"][i]
                sent1 = test["context"][i]
                sent2 = test["context2"][i]
                
                pbar.set_description(f"Predicting {genre}: ({word1} V {word2})")
                
                gold_complexity =  test["binary_comparative_val"][i]
                
                prompt_row = promptLearning_LCP(id_num, word1, word2, sent1, sent2, gold_complexity, model, tokenizer)
                final_dataset.append(prompt_row)
                pbar.update(1)
    
                # print(prompt_row)
        

            
        pbar.close()     
        write_output(final_dataset, f"EN_{genre}_binary_comparative_LCP", model_card)
        final_dataset =[] 

        # input("Continue. Enter?")
