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



def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text


infile= open('dict_all', 'rb')
tfidf_data= pickle.load(infile)

infile= open('norm_doc','rb')
norm_doc = pickle.load(infile)

infile = open('files_index','rb')
files_index = pickle.load(infile)

infile=open('doc_words','rb')
doc_words = pickle.load(infile)

infile=open('doc_freq','rb')
doc_freq = pickle.load(infile)


#df = pd.DataFrame(a, columns=['a'])


def tfidf(query):
    query = sent_tokenize(query)
    text_q = remove_special_characters(str(query))
    text_q = re.sub(r'\d+', '', text_q)
    token_q= word_tokenize(text_q)
    token_q = [word for word in token_q if word not in Stopwords]
    token_q = [word for word in token_q if len(word) > 1]
    token_q = [word.lower() for word in token_q]
    token_q=[word for word in token_q if word in tfidf_data.keys()]
    # creating the and query and vector and calculating its norm
    q_v = []
    q_norm=0
    for w in token_q:
        value= token_q.count(w)* math.log(len(files_index)/doc_freq[w])
        q_v.append(value)
        q_norm += value**2
    q_norm = math.sqrt(q_norm)
    q_v=np.array(q_v)/q_norm
    #caculating the cosine similarity of the query vector with all the documents for the ranking purpose
    cosine_score={}

    for i in range(len(files_index)):
        doc_v=[]
        for w in token_q:
            value=(doc_words[i].count(w)*math.log(len(files_index)/doc_freq[w]))
            doc_v.append(value)
        doc_v=np.array(doc_v)/norm_doc[i]
        cosine_score[i]=np.dot(q_v,doc_v)
    # the score is sorted to bring out the top 5 most similar documents
    score=sorted(cosine_score.items(),key=lambda x:x[1],reverse=True)
    #print(files_index)
    k = 5
    ans= []
    for i in score:
        if k == 0:
            break
        ans.append(files_index[i[0]])
        k-=1
    return ans


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
    solution[k1] = tfidf(q)
# q_rels contains the final output    
Qrels= []
for i in solution:
    for ans in solution[i]:
        Qrels.append ([i,1,ans,1])
        
df = pd.DataFrame(Qrels)
df.to_csv("Qrels_tfidf.txt",header= False, sep = ",",index = False)