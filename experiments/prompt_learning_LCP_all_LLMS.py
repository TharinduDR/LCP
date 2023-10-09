# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:41:02 2023

@author: knorth8
"""
from torch.utils.data import Dataset
import openai        
import csv
import pandas as pd
from tqdm import tqdm # nice loading bar.
from statistics import mode
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, pipeline
import torch
import re
openai.api_key = "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"

   

def promptLearning_LCP(word, sent, gold_complexity):
    

    """ ============================================
        =================== LCP ====================
        ============================================"""
    final_prompts = []

    "> English <"
    prompt1 =f"On a scale from 1 to 5 with 5 being the most difficult, rate the difficulty of the word \"{word}\"?\n"\
            f"Provide a number:"   
            
    prompt2 =f"Sentence: {sent}\n" \
             f"On a scale from 1 to 5 with 5 being the most difficult, rate the difficulty of the word \"{word}\" in the above sentence?\n"\
             f"Provide a number:"
        
        
    prompt3 =f"On a scale from 1 to 10 with 10 being the most difficult, rate the difficulty of the word \"{word}\"?\n"\
            f"Provide a number:"
                              

    prompt4 =f"Sentence: {sent}\n" \
             f"On a scale from 1 to 10 with 10 being the most difficult, rate the difficulty of the word \"{word}\" in the above sentence?\n"\
             f"Provide a number:"
             
    # prompt5 =f"On a scale for 1 to 10 with 10 being the most difficult, rate the difficult of the word \"{word}\" for a child?\n"\
    #          f"Answer:"
               
            
    prompts = [prompt1,prompt2,prompt3,prompt4]

        
    for prompt in prompts:
        final_prompts.append(prompt)


    # print(final_prompts)  
    # input("enter")    
    return final_prompts

class ListDataset(Dataset):

    def __init__(self, original_list):
        self.original_list = [] # clears dataset ready for next genre.
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def normalize_val(val_string, min_val, max_val):
    if val_string.isnumeric() == True:
        # (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        norm_val = (int(val_string) - min_val)/(max_val-min_val)
        # print(f"{prompt1_answer} = {norm_val}")
    
    else:
        # print("Warning: Answer not Int")
        norm_val = 0.0 # returns 0.0 if answer provided by gpt 3.5 is not a number.

    return norm_val

def write_output(final_labels, prompt_name, model_card):
    
    """
        :description: This function writes the extracted data to .tsv files.
        :param data: extracted data to be written to .tsv.
        :return: .tsv files.
    """
    "---- Output ----"

    file_name = f"./prompt_learning_output/Falcon_LCP_results/{model_card}_{prompt_name}.tsv"

    with open(file_name, 'a', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
        tsv_output = csv.writer(f_output, delimiter='\t')
        
        tsv_output.writerow(final_labels)
        
        # print("\n\n==== Dataset Info ====")    
        # print("Output saved as:", file_name)    



if __name__ == '__main__':
    
    # input_df = pd.read_csv("data_debug/v0.02_MultiLex_test.tsv", sep="\t") # used for debugging.
    match_found = 0
    
    # model_card = "TheBloke/Llama-2-7b-chat-fp16"
    # model_card = "tiiuae/falcon-7b-instruct"
    model_card = "mosaicml/mpt-7b-instruct"

    model = model_card
    
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    
    # if torch.cuda.is_available(): # converts to half tensor.
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    pipe = pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
        max_new_tokens=2,
        do_sample=True,
        top_k=1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id)
        
    
    print("GPU found:", torch.cuda.is_available())
    # input("enter")
            
    language_list = ["EN", "PT"]
    test_file_list = ["data/lcp_single_test.tsv", "data/v0.02_MultiLex_test.tsv" ]
    start_at = 0 # select where you want to start process at, either EN or PT.
    
    for i in range(start_at, len(language_list)):
        
        language = language_list[i] # select language.
        test_file = test_file_list[i] # select test file path.
        
        "--- Binary Datasets: Same as Tharindu Datasets --- "
       
        input_df = pd.read_csv(test_file, sep="\t") # CompLex EN test set.     
        test = input_df[["id", "genre", "pt_sentence", "pt_word", "avg_complexity"]] # original headers from MultiLex-P test set.
        
        final_dataset = [] 
        dataset = []
        final_dataset_prompts = []
        
        model_card = model_card.replace("/","-")   
        prompt_name = f"{language}_LCP"
        file_name = f"./prompt_learning_output/Falcon_LCP_results/{model_card}_{prompt_name}.tsv"
        
        with open(file_name, 'a', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
            tsv_output = csv.writer(f_output, delimiter='\t')
            
            headers_output = ["Id","Genre", "Sentence","ComplexWord","Gold_Complexity","prompt1_pred","prompt2_pred","prompt3_pred","prompt4_pred","avg_pred1_2", "avg_pred3_4"]
            
            tsv_output.writerow(headers_output)
            

        with tqdm(total=len(test)) as pbar:
            for i in range(len(test)):
                
                'CompLex: EN'
                sent = test["pt_sentence"][i]
                word =  test["pt_word"][i]
                gold_complexity =  round(test["avg_complexity"][i], 2)

                pbar.set_description(f"Extracting prompts {language}: ({word})")
                
                gold_complexity =  test["avg_complexity"][i]
                
                final_prompts = promptLearning_LCP(word, sent, gold_complexity)
                final_dataset_prompts += final_prompts
                                
                pbar.update(1)
    
                # print(final_prompts) # useful for debugging.
                # input("enter")
                
        pbar.close()   
        
        dataset = ListDataset(final_dataset_prompts) # converts list to dataset ready for pipeline.

        result = pipe(dataset) # gets answers to prompts saved as dict.
        
        print("Result:", result)
        
        with tqdm(total=len(result)) as pbar:
            pbar.set_description(f"Calculating Complexity ({language}: {word})")
            prompt_num = 0
            temp_list_question = []
            temp_list_full_answer = []
            temp_list_answer = []
            index = 0
            for i in result:
                
                id_num = test["id"][index]
                genre = test["genre"][index]
                sent = test["pt_sentence"][index]
                word =  test["pt_word"][index]
                gold_complexity =  round(test["avg_complexity"][index], 2)
                
                prompt_num+=1

                pbar.update(1)
                final_dataset.append(i)
                
                if prompt_num <= 4:
                    prompt1_question = re.sub('Provide a number:.*',"", i[0]['generated_text']).replace("\n","")
                    prompt1_answer = re.findall('Provide a number:.*', i[0]['generated_text']) # returns everything after answer.
                    if type(prompt1_answer) == list:
                        prompt1_answer = prompt1_answer[0].replace("Provide a number:","") # finds and removes prompt and "answer" and returns a string from list. 
                        
                    if prompt1_answer == " ": # captures answers missed by re.
                        prompt1_answer = i[0]['generated_text']
                        
                    temp_list_full_answer.append(prompt1_answer)
 
                    
                    prompt1_answer = prompt1_answer.strip()
                    
                    # print("Answer:", prompt1_answer)
                    # input('enter')
                    
                    if prompt1_answer.isnumeric() == True: # check if number in answer saved as string.
                        # prompt1_answer = int(prompt1_answer) # converts to int. Don't need to do this here. This is done in within the normlize_val function.
  
                        if prompt_num <=2:
                            norm_val1 = normalize_val(prompt1_answer, 1, 5) # normalizes answer (lickert-scale 1 to 5).
                            temp_list_answer.append(norm_val1)
                            temp_list_question.append(prompt1_question)
                            
                        if prompt_num <=4:
                            norm_val1 = normalize_val(prompt1_answer, 1, 10) # normalizes answer (lickert-scale 1 to 10).
                            temp_list_answer.append(norm_val1)
                            temp_list_question.append(prompt1_question)
                            
                    else:
                        temp_list_answer.append(0.5) # appends neteural if no number is generated as answer.

                        
                if prompt_num == 4: # saves to tsv once all prompts have been processed. Needs to == max number of prompts per instance. 
             
                    avg_complexity1 = sum([temp_list_answer[0], temp_list_answer[1]])/2
                    avg_complexity2 = sum([temp_list_answer[2], temp_list_answer[3]])/2

                    prompt_row = [id_num, genre, sent, word, gold_complexity] + [round(temp_list_answer[0],2), round(temp_list_answer[1],2), round(temp_list_answer[2],2), round(temp_list_answer[3],2), round(avg_complexity1, 2), round(avg_complexity2, 2)]
                    
                    write_output(prompt_row, prompt_name, model_card)
                    
                    prompt_num = 0 # resets prompt number. This should be equal to the number of prompts being used.
                    temp_list_full_answer = []
                    temp_list_question = []
                    temp_list_answer = []
                   
               
                    index+=1
                    
                    # print(prompt_row) # very useful for overall debugging.
                    # input("enter")
                    
   
        pbar.close()  
                  
             # clears dataset ready for next genre.

        # input("Continue. Enter?")
