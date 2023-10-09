# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:52:02 2023

@author: knorth8
"""

f = open("PT_binary_comparative_table_results.txt", "r")

row_dic = {}


for row in f:
    split_rows = row.split("&")
    if len(split_rows) > 6:

        row_dic[row] = split_rows[5]
        # print(row_dic)
        # input("enter")
        
        
row_dic = dict(sorted(row_dic.items(), key=lambda item: item[1], reverse=True))


for row in row_dic:
    print(row)