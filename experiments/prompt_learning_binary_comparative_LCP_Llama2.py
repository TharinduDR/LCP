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
        =================== Binaru Comparative LCP ====================
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
    
    prompt7 =f"Which sentence is more difficult: \"{sent1}\" or \"{sent2}\"?\n"\
            f"Answer:"
            
    # prompts = [prompt1,prompt2,prompt3,prompt4,prompt5,prompt6,prompt7]
    prompts = [prompt1,prompt5,prompt7]
        
    for prompt in prompts:
        final_prompts.append(prompt)



    # binary_answer1 = 1
    # binary_answer2 = 1
    # binary_answer3 = 1
    # binary_answer4 = 1
    # binary_answer5 = 1
    # binary_answer6 = 1
    # binary_answer7 = 1
    

    " === for Downloaded Model ==="

    # print(prompt_answers)
       
    # prompt1_answer = pipeline(prompt1)[0]['generated_text'].replace(prompt1, "")
    # if word2 in prompt1_answer and word1 not in prompt1_answer:
    #     binary_answer1 = 0

    # prompt2_answer = pipeline(prompt2)[0]['generated_text'].replace(prompt2, "")
    # if word2 in prompt2_answer and word1 not in prompt2_answer:
    #     binary_answer2 = 0
    
    # prompt3_answer = pipeline(prompt3)[0]['generated_text'].replace(prompt3, "")
    # if word2 in prompt3_answer and word1 not in prompt3_answer:
    #     binary_answer3 = 0

    # prompt4_answer = pipeline(prompt4)[0]['generated_text'].replace(prompt4, "")
    # if word1 in prompt4_answer and word2 not in prompt4_answer:
    #     binary_answer4 = 0

    # prompt5_answer = pipeline(prompt5)[0]['generated_text'].replace(prompt5, "")
    # if word1 in prompt5_answer and word2 not in prompt5_answer:
    #     binary_answer5 = 0
        
    # prompt6_answer = pipeline(prompt6)[0]['generated_text'].replace(prompt6, "")
    # if word1 in prompt6_answer and word2 not in prompt6_answer:
    #     binary_answer6 = 0
    
    # prompt7_answer = pipeline(prompt7)[0]['generated_text'].replace(prompt7, "")
    # if "(b)" in prompt7_answer and "(a)" not in prompt7_answer:
    #     binary_answer7 = 0

            
    # mode_complexity =  mode([binary_answer1, binary_answer2, binary_answer3, binary_answer4, binary_answer5, binary_answer6, binary_answer7])
    # # prompt6_row = [sent, word, gold_complexity] + [avg_complexity, 0]
    
    
    # prompt_row = [word1, word2, sent1, sent2, gold_complexity] + [prompt1_answer, prompt2_answer, prompt3_answer, prompt4_answer, prompt5_answer, prompt6_answer, prompt7_answer,
    #                                                               binary_answer1, binary_answer2, binary_answer3, binary_answer4, binary_answer5, binary_answer6, binary_answer7, mode_complexity]

    # print(prompt_row)
    # input("Enter")

    # pbar.close()            
    return final_prompts

class ListDataset(Dataset):
    
    def __init__(self, original_list):
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

    file_name = f"./prompt_learning_output/Llama_round_2_results/{model_card}_{prompt_name}.tsv"

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
    final_dataset =[] 
    final_dataset_prompts= []
    match_found = 0
    
    model_card = "TheBloke/Llama-2-7b-chat-fp16"
    # model_card = "meta-llama/Llama-2-7b-hf"
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_card)
    model = model_card
    
    # model = LlamaForCausalLM.from_pretrained(model_card)
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    
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
            
    language_list = ["EN"]
    genre_list = ["biomed"]
    
    for language in language_list:
        for genre in genre_list:
        
            "--- Binary Datasets: Same as Tharindu Datasets --- "
            
            test_file = fr"C:\Users\knorth8\Documents\work\complex-pt\LCP\experiments\data_small\binary_comparative_MultiLex\{language}\{genre}\small_binary_comparative_MultiLex_test_{language}.tsv"
            input_df = pd.read_csv(test_file, sep="\t") # CompLex EN test set.
            test = input_df[["id", "word1", "word2", "context", "context2", "binary_comparative_val"]] # CompLex EN test set.
            
            dataset = []
    
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
            
            
            with tqdm(total=len(result)) as pbar:
                pbar.set_description(f"Calculating Complexity")
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
                        if "first sentence" in prompt1_answer.lower() and "second sentence" not in prompt1_answer.lower():
                            binary_answer = 1
                            temp_list_question.append(prompt1_question)
                            temp_list_answer.append(binary_answer)
                            
                        elif word2.lower() in prompt1_answer.lower() and word1.lower() not in prompt1_answer.lower():
                            binary_answer = 0
                            temp_list_question.append(prompt1_question)
                            temp_list_answer.append(binary_answer)
                            
                        else:
                            temp_list_answer.append(1)
                            
                            
                            
                    if prompt_num == 3:
                 
                        mode_complexity =  mode(temp_list_answer)
                        # print("temp_list_full_answer:", len(temp_list_full_answer))
                        # print("temp_list_full_answer:", len(temp_list_answer))
                        
                        prompt_row = [id_num, word1, word2, sent1, sent2, gold_complexity] + [temp_list_full_answer[0], temp_list_full_answer[1], temp_list_full_answer[2], 
                                                                                      temp_list_answer[0], temp_list_answer[1], temp_list_answer[2], mode_complexity]
                        
                        
                        model_card = model_card.replace("/","-")   
                        write_output(prompt_row, f"small_{language}_{genre}_binary_comparative_LCP", model_card)
                        
                        prompt_num = 0 # resets prompt number. This should be equal to the number of prompts being used.
                        temp_list_full_answer = []
                        temp_list_question = []
                        temp_list_answer = []
                       
                        # print(prompt_row)
                   
                        index+=1
     
                        # input("enter")
                        
       
            pbar.close()  
                  
            
            final_dataset = [] 

        # input("Continue. Enter?")
