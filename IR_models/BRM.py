import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
import glob
import re
import os
import numpy as np
import sys
Stopwords = set(stopwords.words('english'))
import pickle
import math
import pandas as pd

infile= open('dict_all', 'rb')
dict_file= pickle.load(infile)

infile = open('files_index','rb')
files_index = pickle.load(infile)


def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text   

  # this function adds the and component to the query in between the words to find the final output vector.
def break1(q_tokens,words):
    for i in range(1,len(q_tokens)):
        if q_tokens[i] not in ["and"]:
            if words[-1] not in ["and"]:
                words.append("and")
                words.append(q_tokens[i])
            else:
                words.append(q_tokens[i])
        elif words[-1] not in ["and"]:
            words.append(q_tokens[i])
    return words

#Function that takes the query and outputs the relevant document for the query according to the boolean IR model
def boolean(query):
    text = remove_special_characters(str(query))
    text = re.sub(r'\d+', '', text)
    q_tokens = word_tokenize(text)
    q_tokens = [word for word in q_tokens if word not in Stopwords]
    q_tokens = [word for word in q_tokens if len(word) > 1]
    q_tokens = [word.lower() for word in q_tokens]

    unique_words = set(dict_file.keys())

    if len(q_tokens)>1:
        words= [q_tokens[0]]
    else:
        words= q_tokens
    # calling the break1 fucntion to add the connectors in the query
    words= break1(q_tokens,words)
    #print(words)

    connectors=[]
    main_q=[]
    # for every query word separating the main word and the connectors
    for w in words:
        if w.lower() in ["and"]:
            connectors.append(w.lower())
        else:
            main_q.append(w)

    n = len(files_index)
    bit_vector = []
    mat = []
    for w in main_q:
        bit_vector = [0]*n
        if w in unique_words:
            for i in dict_file[w].keys():
                bit_vector[i-1]= 1
        mat.append(bit_vector)
    #Doing the 'and' on the the binary vectors containing 1 and 0 for each particular query word 
    for w in connectors:
        v1 = mat[0]
        v2 = mat[1]
        new = [vec1 & vec2 for vec1, vec2 in zip(v1,v2) ]
        mat.pop(0)
        mat.pop(0)
        mat.insert(0,new)

    ans = mat[0]
    ansret=[]
    #print(ans)
    count=0
    l=0
    ans_file= []
    for i in ans:
        if i==1:
            ans_file.append(files_index[l])
        l+=1
    return ans_file
    

# storing the results of the query in the text file
solution={}
def dirback():
    m = os.getcwd()
    n = m.rfind("/")
    d = m[0: n+1]
    os.chdir(d)
    return None

dirback()
query_input = open(sys.argv[1]+".txt","r")
querys =query_input.readlines()
for i in querys:
    #print(i)
    x= i.split("\t")
    k1 =x[0]
    q= x[1]
    solution[k1] = boolean(q)
    
Qrels= []
count=5
for i in solution:
    for ans in solution[i]:
        Qrels.append ([i,1,ans,1])
        count-=1
        if(count==0):
            break
        
df = pd.DataFrame(Qrels)
df.to_csv("Qrels_boolean.txt",header= False, sep = ",",index = False)