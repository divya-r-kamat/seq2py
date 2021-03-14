# seq2py 
  - A transformer based model to translate English text to Python code (with proper whitespace indentations)

<p align="center"><img src="https://user-images.githubusercontent.com/42609155/111054353-fde99480-8491-11eb-8192-cee1f38cf5e6.gif" width="600"></p>


## Data Collection
Collected the data from github and other python sources in the a format such that the description of the code to be written (starts with #), follwed by the code it should generate, and following requirements were taken into consideration:
- we mention whether the code needs to write a program or write a function
- if the code needs to print something, then "print" is mentioned in the text
Here are some examples:
- provide the length of list, dict, tuple, etc
- write a function to sort a list
- write a function to test the time it takes to run a function
- write a program to remove stop words from a sentence provided, etc


## Data cleaning and preprocessing
The training dataset contained around 4600+ examples of English text to python code.

Followed steps were followed to clean and preprocess the data
*  Took the entire dataset and executed the code in python environment to fix any syntactical issues with the code like missing brackets, comma etc
*  To fix the indentation and formating issue used [autopep8](https://pypi.org/project/autopep8/)
for eg: to fix the indentation issue 
    `autopep8 <<filename>> --select=E101,E121 (use  --in-place option to make the changes)`
* Also, used [python_minifier](https://python-minifier.com/) to remove empty lines, annotations and comments.
* Apart from this also had to fix few minor indentation or space issues manually as and when encountered with the failure
* Once the data was cleaned as part of preprocessing, the dataset was divided into English and "python-code" pairs properly
* A custom tokenizer function is been written so that  it generates right tokens and whenever a indent token is encountered it calculates the number of spaces and replaces it with `(number of spaces)/4 *  't'` . 4 is used because we are considering 4 spaces for indentation.


## Different Experiments and Architectures

* [Baseline Model](https://github.com/chinmay-singh/Propaganda/blob/master)
* [Transformer Based Model](https://github.com/chinmay-singh/Propaganda/blob/crf)
* [Transformer Based Model + Pretrained Custom Embedding layer](https://github.com/chinmay-singh/Propaganda/tree/less)
* [Trasnaformer Based Model + Pretrained Custome Embedding + Data Augmentation](https://github.com/chinmay-singh/Propaganda/tree/lexicon)


Experiment | Batch Size | Epoch | Learning Rate | Test Loss |
---|---|---|---|---|
Baseline Model| 16 | 10 | 0.0005| 1.724 |
Transformer Based Model| 28 |20 |0.0003 | 1.438 |
Transformer Based Model + Pretrained Custom Embedding layer| 28 | 20 |0.0003 | 1.486|
Trasnformer Based Model + Pretrained Custome Embedding + Data Augmentation| 16 | 20| 0.003| 1.390 |

## Pretrained Custom Embedding layer

Trained a separate embedding layer for python keywords using glove so that model understands and pays special attention to whitespaces, colon and other things (like comma etc)
- Used CoNala dataset from [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github), to train the glove model.
- Faced some problem using the glove model directly into Pytorch model, so had ro convert glove model to word2vec using glove2word2vec library available in gensim
- Glove model is trained for 50 epochs and to generate embedding vectors of 256 dimension



## Streamlit App

As a end product, two models are choosen [Transformer Based Model](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/model_using_transformer_embedding) and [Transformer Based Model + Pretrained Custom Embedding layer](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/model_with_custom_pretrained_embeddings).  These model are deployed using a Streamlit app, option is provided to choose one of these models to generate the python code.
Streamlit is an open source app framework specifically designed for ML engineers. It allows to create a stunning looking application with only a few lines of code.


## References
https://conala-corpus.github.io/#dataset-information
