
import re
import json
import pickle


import fasttext
from tqdm import tqdm
import pandas as pd
from nltk.tokenize import word_tokenize

white_list=[]
split_word=r'\/|\.|='


def extract_words_from_file(file, min_freq=5):

    words_with_frequency={}

    def add2words(word):
        if word not in words_with_frequency.keys():
            words_with_frequency[word]=1
        else:
            words_with_frequency[word]+=1

    df=pd.read_csv(file)
    
    for _,row in tqdm(df.iterrows(),desc="reading input",total=df.shape[0]):

        ipt=row[0]        

        for w in word_tokenize(ipt):
            # split smaller
            if re.search(split_word,w):
                if w in white_list:
                    add2words(w)
                    continue
                for c in re.split(split_word,w):
                    add2words(c)
                continue
            add2words(w)

    words_filter={}
    for word,freq in words_with_frequency.items():
        if freq > min_freq:
            words_filter[word]=freq
    words=[x[0] for x in sorted(words_filter.items(),key=lambda x:x[1],reverse=True)]


    return words

def extract_sentence_from_file(file):
    
    with open(file[:-4]+"_sentence.txt","w") as f:
        df=pd.read_csv(file)
        for _,row in tqdm(df.iterrows(),desc="reading input",total=df.shape[0]):
            ipt=row[0]
            words=[]
            for w in word_tokenize(ipt):
                # split smaller
                if re.search(split_word,w):
                    if w in white_list:
                        words.append(w)
                        continue
                    for c in re.split(split_word,w):
                        words.append(c)
                    continue
                words.append(w)
            f.write(" ".join(words)+"\n")
    return file[:-4]+"_sentence.txt"

def extract_words_and_embedding_with_wordbag(csv_path):
    print("=============================[word bag]============================")
    print("[1] reading words:")
    words=extract_words_from_file(csv_path)
    print("\t length:",len(words))
    print("[2] encoding words:")
    word_dict={}
    for i,value in enumerate(words):
        print("\t {0: >6} : {1: <30}{2}".format(i+1,value,"\n" if i%4==0 else ""),end="")
        word_dict[value]=i+1
    print("\n[3] storing words:")
    print("\t ",csv_path[:-4]+".json")
    f=open(csv_path[:-4]+".json",'w')
    json.dump(word_dict,f)
    f.close()
    f=open(csv_path[:-4]+".dict",'wb')
    print("\t ",csv_path[:-4]+".dict")
    pickle.dump(word_dict,f)
    f.close()

def extract_words_and_embedding_with_fasttext(csv_path):

    print("=============================[fast text]============================")
    print("[1] reading sentences:")
    text=extract_sentence_from_file(csv_path)
    print("[2] training model...")
    model=fasttext.train_unsupervised(text,model='skipgram',dim=30)
    print(model.words)
    print("[3] saving model...")
    model.save_model(csv_path[:-4]+".ftz")
    print(csv_path[:-4]+".ftz")

def extract_words_and_embedding(binary,type="wordbag"):
    csv_path="./{0}/{0}.csv".format(binary)
    if type=="wordbag":
        extract_words_and_embedding_with_wordbag(csv_path)
    elif type=="fasttext":
        extract_words_and_embedding_with_fasttext(csv_path)
        

if __name__ == "__main__":
    import sys
    extract_words_and_embedding(sys.argv[1],type="fasttext")
