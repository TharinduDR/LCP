# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:41:02 2023

@author: knorth8
"""

import openai        
import csv
import pandas as pd
from tqdm import tqdm # nice loading bar.
import requests
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM

# openai.api_key = "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"
API_TOKEN = "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" # for huggingface API
# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
   

def promptLearning_LCP(genre, sent, word, gold_complexity, model, tokenizer):
    
    """ ============================================
        =================== LCP ====================
        ============================================"""
        
    

    
    "> English <"
    prompt1 =f"On a scale for 0 to 5 with 5 being the most difficult, how difficult is the \"{word}\"?\n"\
            f"Answer:"   
        
    prompt2 =f"On a scale for 0 to 10 with 10 being the most difficult, how difficult is the \"{word}\"?\n"\
            f"Answer:"
                              
    prompt3 =f"Sentence: {sent}\n" \
             f"On a scale for 0 to 5 with 5 being the most difficult, how difficult is the \"{word}\" in the above sentence?\n"\
             f"Answer:"
    
    prompt4 =f"Sentence: {sent}\n" \
             f"On a scale for 0 to 10 with 10 being the most difficult, how difficult is the \"{word}\" in the above sentence?\n"\
             f"Answer:"
             
    prompt5 =f"On a scale for 0 to 10 with 10 being the most difficult, how difficult is the \"{word}\" for a child?\n"\
             f"Answer:"
                       
    def hugging_face_api_query(payload):
    	response = requests.post(API_URL, headers=headers, json=payload)
    	return response.json()
        
    def normalize_val(val_string, min_val, max_val):
        if val_string.isnumeric() == True:
            # (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            norm_val = (int(val_string) - min_val)/(max_val-min_val)
            # print(f"{prompt1_answer} = {norm_val}")
        
        else:
            # print("Warning: Answer not Int")
            norm_val = 0.0 # returns 0.0 if answer provided by gpt 3.5 is not a number.
    
        return norm_val

    # try: # if we lose connection...
    
    " === for Downloaded Model ==="
    inputs = tokenizer(f"{prompt1}",return_tensors="pt")
    prompt1_answer = model.generate(**inputs, max_new_tokens=1)
    prompt1_answer = tokenizer.batch_decode(prompt1_answer, skip_special_tokens=True)[0]
    norm_val1 = normalize_val(prompt1_answer, 0, 5)
    
    inputs = tokenizer(f"{prompt2}",return_tensors="pt")
    prompt2_answer = model.generate(**inputs, max_new_tokens=1)
    prompt2_answer = tokenizer.batch_decode(prompt2_answer, skip_special_tokens=True)[0]
    norm_val2 = normalize_val(prompt2_answer, 0, 10)
    
    inputs = tokenizer(f"{prompt3}",return_tensors="pt")
    prompt3_answer = model.generate(**inputs, max_new_tokens=1)
    prompt3_answer = tokenizer.batch_decode(prompt3_answer, skip_special_tokens=True)[0]
    norm_val3 = normalize_val(prompt3_answer, 0, 5)
    
    inputs = tokenizer(f"{prompt4}",return_tensors="pt")
    prompt4_answer = model.generate(**inputs, max_new_tokens=1)
    prompt4_answer = tokenizer.batch_decode(prompt4_answer, skip_special_tokens=True)[0]
    norm_val4 = normalize_val(prompt4_answer, 0, 10)
    
    
    inputs = tokenizer(f"{prompt5}",return_tensors="pt")
    prompt5_answer = model.generate(**inputs, max_new_tokens=1)
    prompt5_answer = tokenizer.batch_decode(prompt5_answer, skip_special_tokens=True)[0]
    norm_val5 = normalize_val(prompt5_answer, 0, 10)
    
    

    " === for Hugging Face API ==="
    # prompt1_answer = hugging_face_api_query({"inputs": f"{prompt1}",})
    # prompt1_answer = prompt1_answer[0]['generated_text']
    # norm_val1 = normalize_val(prompt1_answer, 0, 5)


    # prompt2_answer = hugging_face_api_query({"inputs": f"{prompt2}",})
    # prompt2_answer = prompt2_answer[0]['generated_text']
    # norm_val2 = normalize_val(prompt2_answer, 0, 10)       
    
    # prompt3_answer = hugging_face_api_query({"inputs": f"{prompt3}",})
    # prompt3_answer = prompt3_answer[0]['generated_text']
    # norm_val3 = normalize_val(prompt3_answer, 0, 5)
    
    # prompt4_answer = hugging_face_api_query({"inputs": f"{prompt4}",})
    # prompt4_answer = prompt4_answer[0]['generated_text']
    # norm_val4 = normalize_val(prompt4_answer, 0, 10)
    
    # prompt5_answer = hugging_face_api_query({"inputs": f"{prompt5}",})
    # prompt5_answer = prompt5_answer[0]['generated_text']
    # norm_val5 = normalize_val(prompt5_answer, 0, 5)
    

    # except:
    #     norm_val1 = 0
    #     norm_val2 = 0
    #     norm_val3 = 0
    #     norm_val4 = 0
    #     norm_val5 = 0
        

    avg_complexity = sum([norm_val1, norm_val2, norm_val3, norm_val4, norm_val5])/5
    
    # print(gold_complexity, norm_val1, norm_val2, norm_val3, norm_val4, norm_val5, round(avg_complexity, 2))
    # input("Check answer")
    
    prompt_row = [genre, sent, word, gold_complexity] + [norm_val1, norm_val2, norm_val3, norm_val4, norm_val5, round(avg_complexity, 2)]
                
    return prompt_row


def write_output(final_labels, prompt_name, model_card):
    
    """
        :description: This function writes the extracted data to .tsv files.
        :param data: extracted data to be written to .tsv.
        :return: .tsv files.
    """
    "---- Output ----"

    file_name = f"./prompt_learning_output/{model_card}_{prompt_name}.tsv"


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
    
    input_df = pd.read_csv("data/v0.02_MultiLex_test.tsv", sep="\t") # MultiLex-PT PT-BR test set.
    test = input_df[["genre", "pt_sentence", "pt_word", "avg_complexity"]] # MultiLex-PT PT-BR test set.
    
    # input_df = pd.read_csv("data/lcp_single_test.tsv", sep="\t") # CompLex EN test set.
    # test = input_df[["corpus", "sentence", "token", "complexity"]] # CompLex EN test set.
      
    
    model_card = "google/flan-t5-base"
    # model_card = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_card)
    
    # model = LlamaForCausalLM.from_pretrained(model_card)
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    
    
    model_card = model_card.replace("/","-")
    
    final_dataset =[] 
    match_found = 0
    
    with tqdm(total=len(test)) as pbar:
        for i in range(len(test)):
            # time.sleep(60)
           
            'MulitLex: PT-BR'
            genre = test["genre"][i]
            sent = test["pt_sentence"][i]
            word =  test["pt_word"][i]
            gold_complexity =  test["avg_complexity"][i]
            
            'CompLex: EN'
            # genre = test["corpus"][i]
            # sent = test["sentence"][i]
            # word =  test["token"][i]
            # gold_complexity =  round(test["complexity"][i], 2)
            
            pbar.set_description(f"Predicting Complexity: (Word: #{word})")

            prompt_row = promptLearning_LCP(genre, sent, word, gold_complexity, model, tokenizer)
            final_dataset.append(prompt_row)
            
            pbar.update(1)

            
    pbar.close()     
    write_output(final_dataset, "PT_prompt_answers_LCP", model_card)
    
