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

   

def promptLearning_LCP(word1, word2, sent1, sent2, gold_complexity, model, tokenizer, pipeline):
    

    """ ===============================================================
        =================== Binary Comparative LCP ====================
        ==============================================================="""
    final_prompts = []

    "> English <"
    prompt1 =f"Which word is more difficult: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"   
        
    # prompt2 =f"Which word is more difficult to read: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
                              
    # prompt3 =f"Which word is more difficult to understand: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
    
    # prompt4 =f"Which word is learned first by a child: \"{word1}\" or \"{word2}\"?\n"\
            # f"Answer:"
             
    prompt5 =f"Which word is less common: \"{word1}\" or \"{word2}\"?\n"\
            f"Answer:"
            
    # prompt6 =f"Which word is more familiar to the average person: \"{word1}\" or \"{word2}\"?\n"\
    #         f"Answer:"
    
    prompt7 =f"Which sentence is more difficult: (a). \"{sent1}\" or (b). \"{sent2}\"?\n"\
            f"Answer:"
            
    # prompts = [prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7]
    prompts = [prompt1,prompt5,prompt7]
        
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



def write_output(final_labels, prompt_name, model_card):
    
    """
        :description: This function writes the extracted data to .tsv files.
        :param data: extracted data to be written to .tsv.
        :return: .tsv files.
    """
    "---- Output ----"

    file_name = f"./prompt_learning_output/MPT-7B_round2_results/{model_card}_{prompt_name}.tsv"

    with open(file_name, 'a', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
        tsv_output = csv.writer(f_output, delimiter='\t')
        
        # headers_output = ["word1", "word2","sent1", "sent2","Gold_Complexity",
        #                   "prompt1_answer", "prompt2_answer", "prompt3_answer", "binary_answer1", "binary_answer2", "binary_answer3", "mode_complexity"]
        
        # tsv_output.writerow(headers_output)
        

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
        max_new_tokens=5,
        do_sample=True,
        top_k=1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id)
        
    
    print("GPU found:", torch.cuda.is_available())
    # input("enter")
            
    language_list = ["PT"]
    genre_list = ["all_genres","bible","biomed", "news"]
    
    for language in language_list:
        for genre in genre_list:
        
            "--- Binary Datasets: Same as Tharindu Datasets --- "
            
            test_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\{language}\{genre}\small_binary_comparative_MultiLex_test_{language}.tsv"
            input_df = pd.read_csv(test_file, sep="\t") # CompLex EN test set.
            test = input_df[["id", "word1", "word2", "context", "context2", "binary_comparative_val"]] # CompLex EN test set.
            
            final_dataset =[] 
            dataset = []
            final_dataset_prompts = []
            
            model_card = model_card.replace("/","-")   
            prompt_name = f"small_{language}_{genre}_binary_comparative_LCP"
            file_name = f"./prompt_learning_output/MPT-7B_round2_results/{model_card}_{prompt_name}.tsv"
            
            with open(file_name, 'a', newline='', encoding='utf-8') as f_output: # for pt-BR encoding is: ISO-8859-1
                tsv_output = csv.writer(f_output, delimiter='\t')
                
                headers_output = ["id", "genre", "word1", "word2","sent1", "sent2","Gold_Complexity",
                                  "prompt1_answer", "prompt2_answer", "prompt3_answer", "binary_answer1", "binary_answer2", "binary_answer3", "mode_complexity"]
                
                tsv_output.writerow(headers_output)
                

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
                    
                    pbar.set_description(f"Extracting prompts {genre}: ({word1} V {word2})")
                    
                    gold_complexity =  test["binary_comparative_val"][i]
                    
                    final_prompts = promptLearning_LCP(word1, word2, sent1, sent2, gold_complexity, model, tokenizer, pipeline)
                    final_dataset_prompts += final_prompts
                                    
                    pbar.update(1)
        
                    # print(prompt_row)
                    
            pbar.close()   
            # ds = system.dataset.toDataSet(['prompts'], [[x] for x in final_dataset_prompts])
            # df = pd.DataFrame({"prompts": final_dataset_prompts})
            
            dataset = ListDataset(final_dataset_prompts)
    
            result = pipe(dataset)
            
            print("Result:", result)
            # input("enter")
            
            
            with tqdm(total=len(result)) as pbar:
                pbar.set_description(f"Calculating Complexity ({genre})")
                prompt_num = 0
                temp_list_question = []
                temp_list_full_answer = []
                temp_list_answer = []
                index = 0
                for i in result:
                    
                    id_num = test["id"][index]
                    word1 = test["word1"][index]
                    word2 = test["word2"][index]
                    sent1 = test["context"][index]
                    sent2 = test["context2"][index]
                    gold_complexity =  test["binary_comparative_val"][index]
                    
                    
                    binary_answer = 1
                    prompt_num+=1
                    # print(i)
                    pbar.update(1)
                    final_dataset.append(i)
                    
                    if prompt_num <= 3:
                        prompt1_question = re.sub('Answer.*',"", i[0]['generated_text']).replace("\n","")
                        prompt1_answer = re.findall('Answer.*', i[0]['generated_text'])
                        if type(prompt1_answer) == list:
                            prompt1_answer = prompt1_answer[0].replace("Answer:","") # finds and removes prompt and "answer" and returns a string from list. 
                            
                        if prompt1_answer == " ": # captures answers missed by re.
                            prompt1_answer = i[0]['generated_text']
                            
                        temp_list_full_answer.append(prompt1_answer)
                        # print("Answer:", prompt1_answer)
                        # input('enter')
                        if "(b)" in prompt1_answer.lower() and "(a)" not in prompt1_answer.lower():
                            binary_answer = 0
                            temp_list_question.append(prompt1_question)
                            temp_list_answer.append(binary_answer)
                            
                        elif word2.lower() in prompt1_answer.lower() and word1.lower() not in prompt1_answer.lower():
                            binary_answer = 0
                            temp_list_question.append(prompt1_question)
                            temp_list_answer.append(binary_answer)
                            
                        else:
                            temp_list_answer.append(1)
                            
                    # print(i) # useful for debugging.
                    # print(prompt1_answer.lower())
                    # print(binary_answer)
                    # input("enter")
                            
                            
                    if prompt_num == 3:
                 
                        mode_complexity =  mode(temp_list_answer)
                        # print("temp_list_full_answer:", len(temp_list_full_answer))
                        # print("temp_list_full_answer:", len(temp_list_answer))
                        
                        prompt_row = [id_num, genre, word1, word2, sent1, sent2, gold_complexity] + [temp_list_full_answer[0], temp_list_full_answer[1], temp_list_full_answer[2], 
                                                                                      temp_list_answer[0], temp_list_answer[1], temp_list_answer[2], mode_complexity]
                        write_output(prompt_row, prompt_name, model_card)
                        
                        prompt_num = 0 # resets prompt number. This should be equal to the number of prompts being used.
                        temp_list_full_answer = []
                        temp_list_question = []
                        temp_list_answer = []
                       
                        # print(prompt_row)
                   
                        index+=1
     
                        # input("enter")
                        
       
            pbar.close()  
                  
             # clears dataset ready for next genre.

        # input("Continue. Enter?")
