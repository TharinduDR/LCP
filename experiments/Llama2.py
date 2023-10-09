# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:49:52 2023

@author: knorth8
"""
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm # nice loading bar.

model = "TheBloke/Llama-2-7b-chat-fp16"

tokenizer = AutoTokenizer.from_pretrained(model)

pipe = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
    max_new_tokens=10,
    do_sample=True,
    top_k=1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)


class ListDataset(Dataset):
    
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]





dataset = ListDataset(["What is the capital of Italy?", "What is the capital of Italy?", "What is the capital of Italy?", "What is the capital of Italy?", "What is the capital of Italy?"])


print(type(dataset))


result = pipe(dataset)

print(result)

for i in result:
    print(i)

# result = [i for i in tqdm(pipeline(dataset))]

# for out in tqdm.tqdm(pipeline()):
#     print(out)




