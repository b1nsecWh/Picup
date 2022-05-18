import re
import torch
import json
import random
import pickle
import os


import fasttext
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader,Dataset
from nltk.tokenize import word_tokenize


class Argv_File_Dataset(Dataset):
    '''
    argv&file input type
    eg. objdump (-A libc-2.27.so)
    '''

    def __init__(self,binary,file_width,argv_min_len=4,type='train'):
        super(Argv_File_Dataset).__init__()
        self.binary         = binary         # binary name
        self.file_width     = file_width     # file encoding width
        self.argv_min_len   = argv_min_len   # arg encoding min width
        self.type           = type           # dataset type (train or test)
        
        self.read_table_data(type)
        self.get_text_vocab_dict()

    def read_table_data(self,type):
        ''' read data from raw csv file '''
        self.table = pd.read_csv("./data/{0}/{0}_{1}.csv".format(self.binary,type))

        self.table_total=len(self.table.values) 
        self.standard_apis=list(self.table.columns)[2:] # apis
        self.standard_api_size=len(self.standard_apis)  
    
    def get_text_vocab_dict(self):
        ''' get arg word dictionary '''

        with open('./data/{0}/params.txt'.format(self.binary),'r') as f:
            lines=f.readlines()
            self.word_dict={}
            self.vocab_size=len(lines)

            for i,line in enumerate(lines):
                self.word_dict[line.strip()]=i+1
    
    def __len__(self):
        return self.table_total

    def __getitem__(self,index):
        # select data
        data=self.table.values[index]

        return self.get_item_tensor(data)
    
    def get_item_tensor(self,data):
        ''' transfer data to tensor'''
        argv_text,file=data[0],data[1]
        
        #=====================argv=====================#
        words=argv_text.split()   # arg split
        word_embed=[]
        for w in words:
            if w in self.word_dict.keys():
                word_embed+=[self.word_dict[w] for _ in range(3)] # enlarge
            else:
                print("unkown params",w)
                self.vocab_size+=1
                self.word_dict[w]=self.vocab_size
                word_embed+=[self.vocab_size  for _ in range(3)]
        # align
        if len(word_embed) <self.argv_min_len: word_embed+=[0 for __ in range(self.argv_min_len-len(word_embed))]
        word_embed=torch.tensor(word_embed)
        
        #======================file=====================#
        file_embed=self.read_file_to_list(file,self.file_width)
        file_embed=torch.tensor(file_embed,dtype=torch.float32)

        #======================api======================#
        apis=np.array(data[2:],dtype=np.int)
        apis=torch.tensor(apis,dtype=torch.float32)
        
        return ([word_embed,file_embed],apis)
    
    def read_file_to_list(self,filename, img_w):
        ''' read file and then tranfer it to a bytes list '''

        with open("./data/{}/files/{}".format(self.binary, filename.split("/")[-1]), "rb") as f:
            content=f.read()
            content=list(content[:img_w * img_w])
            if len(content)<(img_w * img_w): content+=[0 for _ in range((img_w * img_w)-len(content))]
            content=np.reshape(np.array(content), (1, img_w, img_w))
            return content

class Text_Dataset(Dataset):
    '''
    Text type input
    eg. Nginx socket (GET / \n Host:127.0.0.1 .....)
    '''
    white_list = []   
    split_word = r'\/|\.|='

    def __init__(self,binary,min_len=4,type='train',embedding="fasttext"):
        self.binary     = binary
        self.min_len    = min_len
        self.type       = type
        self.embedding  = embedding
    
        self.read_table_data(type)
        self.get_text_vocab_dict(embedding)

    def read_table_data(self,type):
        ''' read table data '''
        self.table = pd.read_csv("./data/{0}/{0}_{1}.csv".format(self.binary,type),header=0,index_col = False)

        self.table_total=len(self.table.values) 
        self.standard_apis=list(self.table.columns)[1:] # apis
        self.standard_api_size=len(self.standard_apis)  
    
    def get_text_vocab_dict(self,embedding):
        if embedding=="fasttext":
            self.fasttext=fasttext.load_model("./data/{0}/{0}.ftz".format(self.binary))
        elif embedding=="wordbag":
            with open('./data/{0}/{0}.json'.format(self.binary),'rb') as f:
                self.wordbag=json.load(f)
                self.word_vocab_size=len(self.word_dict.values())
        else:
            raise NotImplementedError("not sopport embedding %s",embedding)
    
    def __len__(self):
        return self.table_total

    def __getitem__(self,index):
        data=self.table.values[index]
        return self.get_item_tensor(data)

    def get_item_tensor(self,data):
        input,apis=data[0],data[1:]
        
        words=[]
        for w in word_tokenize(input):
            if re.search(self.split_word,w):
                if w in self.white_list:
                    words.append(w)
                    continue
                for c in re.split(self.split_word,w):
                    words.append(c)
                continue
            words.append(w)
        words_embed=[]
        if self.embedding=="fasttext":
            words_embed=[self.fasttext.get_word_vector(x) for x in words]
            if len(words_embed) <self.min_len:words_embed+=[self.fasttext.get_word_vector("") for __ in range(self.min_len-len(words_embed))]
        elif self.embedding=="wordbags":
            words_embed=[self.wordbag[x] if x in self.wordbag.keys() else 0 for x in words]
            if len(words_embed) <self.min_len:words_embed+=[0 for __ in range(self.min_len-len(words_embed))]

        words_embed=torch.tensor(np.array(words_embed))

        apis=np.array(apis,dtype=np.int32)
        apis=torch.tensor(apis,dtype=torch.float32)

        return (words_embed,apis)

if __name__ == "__main__":    
    for step, (X,y) in enumerate(DataLoader(Text_Dataset("nginx"))):
        for i,X in enumerate(X):
            print(i,X,y)