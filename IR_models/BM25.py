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
import pandas as pd

#importing all the libraries required for this model


def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text   

#using the posting list that we have made during the preprocessing phase and other helping documents
# unloading those from the pickle file.
infile= open('dict_all', 'rb')
bm25data= pickle.load(infile)

infile=open('doc_freq','rb')
doc_freq = pickle.load(infile)

infile = open('files_index','rb')
files_index = pickle.load(infile)

infile = open('doc_len','rb')
doc_len = pickle.load(infile)
# average document length for all the documents is calcualted using the doc length of each doc.
avgdoc = 0
for i in doc_len:
    avgdoc+= doc_len[i]
avgdoc = avgdoc/len(files_index)


k= 1.2
b= 0.75
#bm25 scroe is calcualted using the updated term frequency and inverse document frequency formulae and this score
# will be used to rank the documents as the output to the query.
def score(doc):
    score_ans={}
    for i in range(len(files_index)):
        score_ans[i]= 0
    # the preprocessing steps over the query texts
    query = remove_special_characters(doc)
    que = re.sub(re.compile('\d'),'',query)
    words = word_tokenize(que)
    words = [word.lower() for word in words]
    ps = PorterStemmer()
    words=[ps.stem(word) for word in words]
    words=[word for word in words if word not in Stopwords]
    #print(words)
    # calcualting scores for all the documents
    for i in range(len(files_index)):
        score_ans[i]=0
        for q in words:
            tf=0
            if q in bm25data:
                if i in bm25data[q]:
                    tf=bm25data[q][i]
            idf1=0
            if q in doc_freq:
                idf1= doc_freq[q]
            n= len(files_index)
            idf = math.log((n-idf1+0.5)/(idf1+0.5))
            #value here contains the final score for the document storing it in the score array
            value=idf*(k+1)*tf/(tf+k*(1-b+b*(doc_len[i]/avgdoc)))
            score_ans[i]+= value
    return score_ans
            
# the function to create a bm25 ranking of dcuments for a query.
#here top 5 documents will be extracted with the heighest scores, count=5
def bm25(query):
    ans= []
    #score_ans={}
    
    score_ans = score(query)
    # sorting on the basis of scores.
    score_final= sorted(score_ans.items(),key = lambda x: x[1],reverse= True)
    count=5

    for i in score_final:
        if count == 0:
            break
        ans.append(files_index[i[0]])
        count-=1
    return ans

    
 
# Taking the file as input and processing the queries in the file and storing the output in the text file
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
    x= i.split("\t")
    k1 =x[0]
    q= x[1]
    solution[k1] = bm25(q)
    
Qrels= []
for i in solution:
    for ans in solution[i]:
        Qrels.append([i,1,ans,1])
        
df = pd.DataFrame(Qrels)
df.to_csv("Qrels_bm25.txt",header= False, sep = ",",index = False)