import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import re
import os
import numpy as np
import math
import sys
Stopwords = set(stopwords.words('english'))
import pickle
from collections import Counter


def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text   

def unique_word(words):
    st=[]
    freq={}
    for w in words:
        if w not in st:
            st.append(w)
    for w in st:
        freq[w]= words.count(w)
    return freq



    

files= 'english-corpora/*'
dict_all={}
files_index={}
idx=0
doc_freq={}
norm_words= []
dict_global={}
doc_len={}
ps = PorterStemmer()
for file in glob.glob(files):
    #print(idx)
    name = file
    file = open(file , "r",encoding='UTF-8')
    text = file.read()
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [ps.stem(word) for word in words]
    words = [word for word in words if word not in Stopwords]
    norm_words.append(words)
    dict_global.update(unique_word(words))           
   # dict_all.update(unique_word(tokenized_words))
    files_index[idx] = os.path.basename(name)
    idx = idx + 1
    

unique_words_all = set(dict_global.keys())  
#print(dict_all)



    

dict_all={}
doc_freq={}
for i in unique_words_all:
    dict_all[i]={}
    doc_freq[i]=0

id= 0  
for file in glob.glob(files):
    #print(id)
    name = file
    file = open(file , "r",encoding="utf8")
    text = file.read()
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    doc_len[id]=len(words)
    words=[word for word in words if word not in Stopwords]
    
    counter=Counter(words)
    for i in counter.keys():
        doc_freq[i]=doc_freq[i]+1
        dict_all[i][id]=counter[i]   
    id=id+1

    
    

    
    
outfile=open('doc_freq','wb')
pickle.dump(doc_freq,outfile)
outfile.close()


outfile= open('dict_all','wb')
pickle.dump(dict_all,outfile)
outfile.close()    

outfile=open('files_index','wb')
pickle.dump(files_index,outfile)
outfile.close()
    
outfile=open('doc_len','wb')
pickle.dump(doc_len,outfile)
outfile.close()  

norm_doc={}
id =0
for i in norm_words:
    val=0
    for j in set(i):
        val+=(i.count(j)*math.log(len(files_index)/doc_freq[j]))**2
    norm_doc[id]=(math.sqrt(val))
    #print(id)
    id+=1



outfile= open('norm_doc','wb')
pickle.dump(norm_doc,outfile)
outfile.close()  

outfile= open('doc_words','wb')
pickle.dump(norm_words,outfile)
outfile.close()  
