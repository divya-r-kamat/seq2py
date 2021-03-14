
import torch
from torchtext import data 
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import unicodedata
import codecs
import csv
import time
import random
import re
import os
import pickle
from io import open
import itertools
import math
import pandas as pd
from torchtext.legacy.data import Field, BucketIterator, LabelField, TabularDataset
import streamlit as st

BASE_DIR='/content/drive/MyDrive/seq2py'
#https://stackoverflow.com/questions/62922640/calling-a-function-in-a-different-python-file-using-google-colab
# %cd $BASE_DIR/utils
import utils
from utils import preprocess, helper,transformer
from utils.transformer import Encoder,Decoder,Seq2Seq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Generate python code from english text")

with open(os.path.join(BASE_DIR ,"model/model1_src_stio.pkl"),"rb") as f:
  src_stoi = pickle.load(f)
with open(os.path.join(BASE_DIR ,"model/model1_src_itos.pkl"),"rb") as f:
  src_itos = pickle.load(f)


with open(os.path.join(BASE_DIR ,"model/model1_trg_stio.pkl"),"rb") as f:
  trg_stoi = pickle.load(f)
with open(os.path.join(BASE_DIR ,"model/model1_trg_itos.pkl"),"rb") as f:
  trg_itos = pickle.load(f)

INPUT_DIM = len(src_stoi)
OUTPUT_DIM = len(trg_stoi)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 4
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.15
DEC_DROPOUT = 0.15

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)


dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = src_stoi['<pad>']
TRG_PAD_IDX = trg_stoi['<pad>']

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


 

@st.cache(suppress_st_warning=True,ttl=1000)
def load_model(model_path):
    # Load model
    if (status == 'Seperate Glove Embedding'): 
        trained_model = os.path.join(model_path,"model/tut6-model_20210311_run4_glove_moredata.pt")
    else:
        trained_model = os.path.join(model_path,"model/tut6-model_20210311_run2.pt")
    model.load_state_dict(torch.load(trained_model));
    return model

# radio button 
# first argument is the title of the radio button 
# second argument is the options for the ratio button 
status = st.radio("Select Model: ", ('Transformer Model - Custom Pretrained Embedding', 'Transformer Model - Embedding Layer'))

sentence = st.text_input("Enter english text:")
if(st.button("Generate code")): 
    if sentence is not None:
          model=load_model(BASE_DIR)
          model.eval()
          output_words,_ = helper.translate_sentence(sentence,src_stoi,trg_stoi,trg_itos, model, device)
          # Format and print response sentence
          output_words[:] = [x for x in output_words if not (x == '<eos>' or x == '<pad>')]
          output_words.insert(0, '')
          st.code('Python Code Generated:\n' + ' '.join(output_words))
          