import torch
from torch.utils.data import Dataset , DataLoader 
import tiktoken



class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #A
        for i in range(0, len(token_ids) - max_length, stride): #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): #C
        return len(self.input_ids)
    def __getitem__(self, idx): #D
         return self.input_ids[idx], self.target_ids[idx]
    




def create_dataloader_v1(txt, batch_size=4,
    max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader




# filename =  "/home/saaho/project-7 /Research/coding/Gpt/combined_qa.txt"
# with open(filename , 'r') as file:
#     file_content =  file.read()

# print("File Content Loaded")

# train_dataloader =  create_dataloader_v1(txt=file_content , batch_size= 512 , shuffle= True ,drop_last=True  ,stride=128 )
# print(train_dataloader)