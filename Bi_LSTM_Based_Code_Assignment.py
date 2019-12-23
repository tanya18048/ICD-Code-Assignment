
# coding: utf-8

# In[3]:


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


# In[63]:


#Importing the pubmed embeddings
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('PMC-w2v.bin', binary=True)


# In[6]:


#Creating dataframe of all discharge summaries
df = pd.read_csv("NOTE_EVENTS.csv")


# In[7]:


#Creating dataframe of the corresponding ICD9 labels
df_icd = pd.read_csv("DIAGNOSES_ICD.csv")


# In[3]:


#Creating dataframe with descriptions of all ICD9 codes
df_desc = pd.read_csv("icd9_3digit.csv")


# In[9]:


#Creating dictionary containing notes text clubbed by HADM IDs 
#The text is first filtered for the diagnosis section and then added to the dictionary
df_dict={}
ctr=0
for idx, row in df.iterrows():
    ctr+=1
    hadm_id = row['HADM_ID']
    if(hadm_id!='nan'):
        if(hadm_id not in df_dict.keys()):
            df_dict[hadm_id]=[]
        text=row['TEXT']
        index1 = text.rfind('Final Diagnosis')
        index2 = text.rfind('Discharge Diagnosis')
        if(index1!=-1 or index2!=-1):
            if(index1!=-1):
                text = text[index1:]
        else:
                text = text[index2:]
        df_dict[hadm_id].append(text)
    print(ctr, end='\r')


# In[13]:


#Creating dictionary containing ICD labels clubbed by HADM IDs
df_icd_dict={}
ctr=0
for idx, row in df_icd.iterrows():
    ctr+=1
    hadm_id = row['HADM_ID']
    if(hadm_id!='nan'):
        if(hadm_id not in df_icd_dict.keys()):
            df_icd_dict[hadm_id]=[]
        df_icd_dict[hadm_id].append(row['ICD9_CODE'])
    print(ctr, end='\r')


# In[14]:


filehandler = open("df_icd_dict", 'wb') 
pickle.dump(df_icd_dict, filehandler)


# In[15]:


#Pre-processing the summary text
for key, value in df_dict.items():
    text=str(re.sub('[^a-zA-Z \n\.]', '', str(value)))
    text=re.sub('[\n]', ' ', str(text))
    text=re.sub('n n n n n n n n n n n n n n n n n n n n n n n n n n', '', str(text))
    df_dict[key] = sent_tokenize(text)


# In[16]:


filehandler = open("df_dict", 'wb') 
pickle.dump(df_dict, filehandler)


# In[4]:


#Creating the dictionary containing all tokens in the ICD codes
codes_dict={}
num=0
for text in df_desc['Long Description']:
    text=str(re.sub('[^a-zA-Z \n\.]', '', str(text)))
    text=re.sub('[\n]', ' ', text)
    codes_dict[num]=list(sent_tokenize(text))
    num+=1


# In[5]:


filehandler = open("codes_dict", 'wb') 
pickle.dump(codes_dict, filehandler)


# In[53]:


#################################################################################################################################
#Execution of code starts from here
filehandler = open("df_dict", 'rb') 
df_dict = pickle.load(filehandler)


# In[13]:


filehandler = open("codes_dict", 'rb') 
codes_dict = pickle.load(filehandler)


# In[14]:


#Converting text of summaries to pubmed embeddings
all_sentences=[]
ctr=0
for key, value in df_dict.items():
    print(ctr, end='\r')
    ctr+=1
    for sent in value:
        s=[]
        w_s=word_tokenize(sent)
        for word in w_s:
            try:
                s.append(model[word])
            except:
                pass
        all_sentences.append(s)


# In[7]:


#Converting text of code descriptions to pubmed embeddings
ctr=0
for key, value in codes_dict.items():
    print(ctr, end='\r')
    ctr+=1
    for sent in value:
        s=[]
        w_s=word_tokenize(sent)
        for word in w_s:
            try:
                s.append(model[word])
            except:
                pass
        all_sentences.append(s)


# In[15]:


del df_dict
del codes_dict
del model


# In[64]:


def convert_to_pubmed(sent):
    w_s=word_tokenize(sent)
    s=[]
    for word in w_s:
        try:
            s.append(model[word])
        except:
            pass
    return s


# In[65]:


def process_length(sent):
    len_append = 50 - len(sent)
    arr = np.zeros(200)
    for i in range(len_append):
        sent.append(arr)
    return np.asarray(sent[:50])


# In[42]:


def data_generator(all_sentences, batch_size):
    n_batches=len(all_sentences)//batch_size
    n_sentences=len(all_sentences)
    for idx in range(0, n_sentences, batch_size):
        section=all_sentences[idx:idx+batch_size]
        batch=[]
        for sent in section:
            sent = convert_to_pubmed(sent)
            sent = process_length(sent)
            batch.append(sent)  
        batch = np.asarray(batch)
        yield (batch, batch)


# In[37]:


#Use BiLSTM Encoder with Masking
inp = Input(shape=(50, 200))
masking = Masking(mask_value=np.zeros(200))(inp)

encoded = Bidirectional(LSTM(150))(masking)

decoded = RepeatVector(50)(encoded)
decoded = LSTM(200, return_sequences=True)(decoded)

encoder_decoder = Model(inp, decoded)
encoder = Model(inp, encoded)
encoder_decoder.summary()

encoder_decoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


#Training the model for 5 epochs
history = encoder_decoder.fit(all_sents1, all_sents1, batch_size=32, validation_split=0.2 epochs=5, shuffle=True)


# In[40]:


steps_per_epoch = len(all_sentences)//32


# In[47]:


encoder_decoder.fit_generator(data_generator(all_sentences, 32), steps_per_epoch=steps_per_epoch, epochs=1)
encoder_decoder.fit_generator(data_generator(all_sentences, 32), steps_per_epoch=steps_per_epoch, epochs=1)


# In[50]:


#Stand-alone auto encoder
auto_encoder = Model(inputs=encoder_decoder.inputs, outputs=encoded)


# In[51]:


#saving the models
filehandler = open("autoencoder", 'wb') 
pickle.dump(auto_encoder, filehandler)
filehandler = open("encoder_decoder", 'wb') 
pickle.dump(encoder_decoder, filehandler)


# In[ ]:


#Encoding the sentences
#Use df_dict, codes_dict
encoded_summaries={}
ctr=0
for key, text in df_dict.items():
    text_=[]
    for sent in text:
        sent = convert_to_pubmed(sent)
        sent = process_length(sent)
    text_.append(sent)
    encoded_summaries[key] = encoder.predict(np.asarray(text_))
    
    ctr+=1
    print(ctr, end='\r')


# In[ ]:


#Encoding the 1200 codes
encoded_codes={}
ctr=0
for key, text in codes_dict.items():
    text_=[]
    for sent in text:
        sent = convert_to_pubmed(sent)
        sent = process_length(sent)
    text_.append(sent)
    encoded_codes[key] = encoder.predict(np.asarray(text_))
    
    ctr+=1
    print(ctr, end='\r')


# In[ ]:


#Computing max_100 codes using cosine similarity
max_100={}
for key1, value1 in encoded_summaries.items():
    print(i, end="\r")
    max_100[i]=dict()
    for x in range(0,100):
        max_100[i][x]=0
    for key2, value2 in codes_encoded.items():
        max_cs=0
        for i in range(len(summaries_encoded[key1])):
            cos_sim = np.dot(codes_encoded[key2][0], summaries_encoded[key1][i])/(norm(codes_encoded[key2][0])*norm(summaries_encoded[key1][i]))
            max_cs = max(max_cs, cos_sim)
        min_100 = min(max_100[key1], key=max_100[key1].get)
        if(max_100[key1][min_100]<max_cs):
            del max_100[key1][min_100]
            max_100[key1][key2]=max_cs



