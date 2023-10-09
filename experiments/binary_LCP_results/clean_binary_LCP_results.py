# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:30:48 2023

@author: knorth8
"""

import re


file_results = open("small/PT_xlm-roberta-base_all_results.txt", "r", encoding="utf-8") # opens tsv file.


# print(file_results)



scores = "&& xlm-roberta-base & " 
count= 0 

# print(file_results)

for row in file_results:
    
    
    # print(row)
    # input("enter")
    if "=" in row:
    
       score_type = re.findall(r'\w*=', row) # returns all instances of a $ and its preceding numbers  
       
       score = re.findall(r'\d.\d*', row) # returns all instances of a $ and its preceding numbers  
              
       score_type = [x.replace("=", "") for x in score_type]
       score_type = [i for i in score_type if i]
       
       if len(score_type) > 0:
           count +=1
           scores+=str(round(float(score[0]), 4))+" & "
       if count == 16:
           scores+=" \\\\ &&& "
           count = 0
           
           
           
print(scores)
