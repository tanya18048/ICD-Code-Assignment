
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from keras.models import Model
from keras.layers import LSTM, Dense, TimeDistributed, Input, Masking, RepeatVector, Bidirectional
import pickle


# In[4]:


stop_words = stopwords.words('english')


# In[5]:


#Creating dataframe of all discharge summaries
df = pd.read_csv("NOTE_EVENTS.csv")


# In[6]:


#Creating dataframe of the corresponding ICD9 labels
df_icd = pd.read_csv("DIAGNOSES_ICD.csv")


# In[7]:


#Creating dataframe with descriptions of all ICD9 codes
df_desc = pd.read_excel("icd9_codes.xlsx", index=False)


# In[8]:


#Dataframe containing only the text of discharge summaries
all_discharge_summaries = df[['ROW_ID','TEXT']]


# In[10]:


#Extracting the final diagnosis or discharge diagnosis sections from all the summaries
count=0
shortened_summaries = pd.DataFrame()
for i, discharge_summary in all_discharge_summaries.iterrows():
    text = discharge_summary.loc['TEXT']
#     print("-------------")
#     print(text)
    index1 = text.rfind('Final Diagnosis')
    index2 = text.rfind('Discharge Diagnosis')
    if(index1!=-1 or index2!=-1):
        #print(index1, index2)
        count+=1
        if(index1!=-1):
            shortened_summaries = shortened_summaries.append(pd.Series([i, text[index1:]]), ignore_index=True)
        else:
            shortened_summaries = shortened_summaries.append(pd.Series([i, text[index2:]]), ignore_index=True)
    if(i%1000==0):
        print(i, end="\r")
print(count)


# In[ ]:


#Saving shortened_summaries as csv
shortened_summaries.to_csv('shortened_summaries.csv') 


# In[18]:


#Creating data structures containing the processed text
all_summary_text=[]
summaries_dict={}
for row in shortened_summaries.iterrows():
    text=row[1]
    index=text[0]
    text=str(re.sub('[^a-zA-Z \n\.]', '', str(text[1])))
    text=re.sub('[\n]', ' ', str(text))
    all_summary_text.append(sent_tokenize(text))
    summaries_dict[index] = sent_tokenize(text)


# In[19]:


#Creating the dictionary containing all tokens in the ICD codes
codes_dict={}
num=0
for text in df_desc['LONG DESCRIPTION']:
    #text=row[1]
    text=str(re.sub('[^a-zA-Z \n\.]', '', str(text)))
    text=re.sub('[\n]', ' ', text)
    codes_dict[num]=list(sent_tokenize(text))
    num+=1


# In[20]:


#List of all code sentences
all_codes_text=list(codes_dict.values())


# In[21]:


#Combining the lists of summary texts and codes text
all_text = all_summary_text + all_codes_text


# In[22]:


#Importing the pubmed embeddings
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('PMC-w2v.bin', binary=True)


# In[23]:


#Initial encoding of data with Pubmed Embeddings
all_sents_pubmed = []
for text in all_text:
    pubmed_text=[]
    for sent in text:
        pubmed_sent=[]
        for word in word_tokenize(sent):
            try:
                word_pubmed = model[word]
                pubmed_sent.append(word_pubmed)
            except:
                pass
        pubmed_text.append(pubmed_sent)
    all_sents_pubmed.append(pubmed_text)


# In[24]:


#Restricting size of every summary to 50 sentences
max_len=50

all_sents = []
for text in all_sents_pubmed:
    for sent in text:
        if(len(sent)<50):
            len_append = max_len - len(sent)
            arr = np.zeros(200)
            for i in range(len_append):
                sent.append(arr)
        all_sents.append(np.asarray(sent[:50]))   


# In[25]:


#Converting list to np array
all_sents1 = np.asarray(all_sents)


# In[35]:


#Use BiLSTM Encoder with Masking
inp = Input(shape=(50, 200))
masking = Masking(mask_value=np.zeros(200))(inp)

encoded = Bidirectional(LSTM(150))(masking)

decoded = RepeatVector(50)(encoded)
decoded = LSTM(200, return_sequences=True)(decoded)

sequence_autoencoder = Model(inp, decoded)
encoder = Model(inp, encoded)
sequence_autoencoder.summary()

sequence_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[36]:


#Training the model for 5 epochs
history = sequence_autoencoder.fit(all_sents1, all_sents1, batch_size=32, epochs=5, shuffle=True)


# In[37]:


#Stand-alone auto encoder
auto_encoder = Model(inputs=sequence_autoencoder.inputs, outputs=encoded)


# In[38]:


#saving the models
filehandler = open("autoencoder", 'wb') 
pickle.dump(auto_encoder, filehandler)
filehandler = open("sequence_autoencoder", 'wb') 
pickle.dump(sequence_autoencoder, filehandler)


# In[39]:


#Creating dictionary containing encoded summaries
summaries_encoded={}
for key, value in summaries_dict.items():
    print(key, end='\r')
    summaries_encoded[key]=[]
    for sent in value:
        pubmed_sent=[]
        for word in word_tokenize(sent):
            try:
                word_pubmed = model[word]
                pubmed_sent.append(word_pubmed)
            except:
                pass
        if(len(pubmed_sent)<50):
            len_append = max_len - len(pubmed_sent)
            #print(len_append)
            arr = np.zeros(200)
            for i in range(len_append):
                pubmed_sent.append(arr)
        x = np.asarray(pubmed_sent[:50])
        x = x.reshape(1, 50, 200)
        x_pred = auto_encoder.predict(x)
        summaries_encoded[key].append(x_pred)


# In[45]:


#Saving the encoded summaries as a pickle object
filehandler = open("summaries_encoded", 'wb') 
pickle.dump(summaries_encoded, filehandler)


# In[43]:


#Creating dictionary containing encoded ICD codes
codes_encoded={}
for key, value in codes_dict.items():
    print(key, end='\r')
    codes_encoded[key]=[]
    for sent in value:
        pubmed_sent=[]
        for word in word_tokenize(sent):
            try:
                word_pubmed = model[word]
                pubmed_sent.append(word_pubmed)
            except:
                pass
        if(len(pubmed_sent)<50):
            len_append = max_len - len(pubmed_sent)
            #print(len_append)
            arr = np.zeros(200)
            for i in range(len_append):
                pubmed_sent.append(arr)
        x = np.asarray(pubmed_sent[:50])
        x = x.reshape(1, 50, 200)
        x_pred = auto_encoder.predict(x)
        codes_encoded[key].append(x_pred)


# In[44]:


#Saving the encoded codes as pickle object
filehandler = open("codes_encoded", 'wb') 
pickle.dump(codes_encoded, filehandler)


# In[ ]:


#Compute the cosine similarity results

#Assigning codes if similarity>0.8

code_assignment={}
for key1, value1 in summaries_encoded.items():
    code_assignment[key1]=[]
    print(key1, end='\r')
    for key2, value2 in codes_encoded.items():
        max_cs=0 #To store cosine similarity value of a code with the sentence it matches the most
        for i in range(len(summaries_encoded[key1])):
            cos_sim = np.dot(codes_encoded[key2][0][0], summaries_encoded[key1][i][0])
            norm = np.linalg.norm(codes_encoded[key2][0][0])*np.linalg.norm(summaries_encoded[key1][i][0])
            cos_sim = cos_sim/norm
            max_cs = max(max_cs, cos_sim)
        if(max_cs>0.8):
            code_assignment[key1].append(key2)


# In[ ]:


#Assigning the top 1000 codes

max_1000={}
for key1 in summaries_encoded.items():
    print(i, end="\r")
    max_1000[i]=dict()
    for x in range(0,1000):
        max_1000[i][x]=0
    for key2 in codes_encoded.items():
        max_cs=0
        for i in range(len(summaries_encoded[key1])):
            cos_sim = np.dot(codes_encoded[key2][0], summaries_encoded[key1][i])/(norm(codes_encoded[key2][0])*norm(summaries_encoded[key1][i]))
            max_cs = max(max_cs, cos_sim)
        min_1000 = min(max_1000[key1], key=max_1000[key1].get)
        if(max_1000[key1][min_1000]<max_cs):
            del max_5[key1][min_1000]
            max_1000[key1][key2]=max_cs


# In[ ]:


#Computing the results

tp=0
tn=0
fp=0
fn=0


# In[ ]:


#Converting the dataframe containing the assigned codes to dictionary
df_icd_dict=dict()
for idx,row in df_icd.iterrows():
    row_id = row[0]
    code = df_icd.iloc[idx]['ICD9_CODE']
    df_icd_dict[row_id] = code

