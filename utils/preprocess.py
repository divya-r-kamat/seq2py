import pandas as pd
import tokenize
from io import open
import io
import re
import spacy
spacy_en = spacy.load('en')


def read_data(file_name):
    datasets = [[]]
    #file_name = '/content/drive/MyDrive/seq2py/new_file.py'

    with open(file_name) as f:
      #my_dict = {"description":[],"code":[]}
      for line in f:
        if line.startswith('#'):
          comment = line.split('\n#')
          if datasets[-1] != []:
            # we are in a new block
            datasets.append(comment)
        else:
          stripped_line = line#.strip()
          if stripped_line:
            datasets[-1].append(stripped_line)
    
    return datasets
    
    
def tranform_to_dataframe(datasets):
    raw_data = {'Description' : [re.sub(r"^#(\d)*(.)(\s)*",'',x[0]).strip() for x in datasets], 'Code': [''.join(x[1:]) for x in datasets]}
    df = pd.DataFrame(raw_data, columns=["Description", "Code"])
    df['Code'].replace("", float("NaN"), inplace=True)
    df.dropna(subset = ["Code"], inplace=True)
    return df
    
def train_val_split(df):
    '''Dividing the data into train and validation dataset'''
    train_df = df.sample(frac = 0.80) 
    # Creating dataframe with rest of the 20% values 
    valid_df = df.drop(train_df.index)
    return train_df,valid_df
    
    
def tokenize_python(code_string):
    """
    Tokenizes python code snippet into a list of tokens
    """
    tokens = []
    for toknum, tokval, start, end, line  in tokenize.generate_tokens(io.StringIO(code_string).readline):
        #tokname = py_token_name[toknum]
        if toknum == tokenize.INDENT:
            val = int(len(tokval)/4)
            tokens.append(val*'\t')
        else:
            tokens.append(tokval)
    return tokens


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
    
    

def tokenize_python_code(code_string):
    """
    Tokenizes python code snippet into a list of tokens
    """
    tokens = []
    for toknum, tokval, start, end, line  in tokenize.generate_tokens(io.StringIO(code_string).readline):
        indent_val=len(re.findall(r"^ *", line)[0])
        val = int(indent_val/4)
        if toknum != tokenize.INDENT:
          if start[1] == indent_val:
            if tokval != "":
              tokens.append(val*'\t')
              tokens.append(tokval)
          else:
            tokens.append(tokval)
    return tokens
