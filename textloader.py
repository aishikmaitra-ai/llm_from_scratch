import os
import re
import urllib.request
import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken

#Importing our Data File
def dataset_load(urled:str):
    if not os.path.exists("story.txt"):
        # url=("https://en.wikisource.org/wiki/The_Verdict")
        url=urled
        file_path="story.txt"
        urllib.request.urlretrieve(url,file_path)
    with open("story.txt","r",encoding="utf-8") as f:
        raw_text=f.read()
    return raw_text

#Split using basic words and punctuations
def splitter(text):
    r=[]
    result=re.split(r'([,.:;?_!"()\']|--|\s)',text)
    result=[item.strip() for item in result if item.strip()]
    preprocessed=result
    all_words=sorted(len(preprocessed))
    vocab={token:integer for integer,token in enumerate(all_words)}

    return vocab

#Tokenizer based Class
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}
    def encode(self,text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        # preprocessed=[
        #     item if item in self.str_to_int
        #     else "<|unk|>" for item in preprocessed
        # ]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids): 
        text=" ".join([self.int_to_str[i] for i in ids])
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text 
class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={i:s for s,i in vocab.items()}
    def encode(self,text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        preprocessed=[
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text=" ".join([self.int_to_str[i] for i in ids])
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text 
    
def ownworker(txt):
    vocab=splitter(txt)
    tokenizer=SimpleTokenizerV2(vocab)
    #Encoder
    ids=tokenizer.encode(txt)
    #Decoder
    dec=tokenizer.decode(ids)
    return ids,dec

def tiktoken_func(text):
    content_size=4
    tokenizer=tiktoken.get_encoding("gpt2")
    enc_text=tokenizer.encode(text)
    #work with sample
    enc_sample=enc_text[50:]
    x=enc_sample[:content_size]
    y=enc_sample[1:content_size+1]

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids=[]# Initiate Token Ids and Target Ids
        self.target_ids=[]

        token_ids=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,ids):
        return self.input_ids[ids],self.target_ids[ids]

def create_dataloader_v1(txt,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    tokenizer=tiktoken.get_encoding("gpt2")
    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
def mydata(text):
    vocab_size=50257
    output_dims=256
    max_length=4
    context_length=max_length

    token_embedding_layer=torch.nn.Embedding(vocab_size,output_dims)
    dataloader=create_dataloader_v1(text,batch_size=8,stride=4,max_length=4,shuffle=False)
    data_iter=iter(dataloader)
    inputs,targets=next(data_iter)
    #Token embeddings
    token_emb=token_embedding_layer(inputs)
    pos_embed_layer=torch.nn.Embedding(context_length,output_dims)
    pos_emb=pos_embed_layer(torch.arange(max_length))
    #Coding the input embeddings
    input_embs=token_emb+pos_emb

    return input_embs

    
