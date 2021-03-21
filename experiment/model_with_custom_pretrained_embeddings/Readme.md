# [Transformer Based Model + Pretrained Custom Embedding layer](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/model_with_custom_pretrained_embeddings)

[Link to Training Notebook](https://github.com/divya-r-kamat/seq2py/blob/main/experiment/model_with_custom_pretrained_embeddings/Translate_English_text_to_python_training_Glove_model.ipynb)

## Pretrained Custom Embedding layer [(Notebook Link)](https://github.com/divya-r-kamat/seq2py/blob/main/experiment/model_with_custom_pretrained_embeddings/training_glove_word2vec_embedding.ipynb)

Trained a separate embedding layer for python keywords using glove so that model understands and pays special attention to whitespaces, colon and other things (like comma etc)
- Used CoNala dataset from [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github), to train the glove model.
- Faced some problem using the glove model directly into Pytorch model, so had ro convert glove model to word2vec using glove2word2vec library available in gensim
- Glove model is trained for 50 epochs and to generate embedding vectors of 256 dimension

### Freezing and Unfreezing Embeddings
The model is trained for 20 epochs. During the first 15 epochs we are going to freeze the weights (parameters) of our embedding layer. For the last 5 epochs we'll allow our embeddings to be trained.

Why would we ever want to do this? Sometimes the pre-trained word embeddings we use will already be good enough and won't need to be fine-tuned with our model. If we keep the embeddings frozen then we don't have to calculate the gradients and update the weights for these parameters, giving us faster training times. This doesn't really apply for the model used here, but we're mainly covering it to show how it's done. Another reason is that if our model has a large amount of parameters it may make training difficult, so by freezing our pre-trained embeddings we reduce the amount of parameters needing to be learned.

To freeze the embedding weights, we set model.embedding.weight.requires_grad to False. This will cause no gradients to be calculated for the weights in the embedding layer, and thus no parameters will be updated when optimizer.step() is called.

Then, during training we check if FREEZE_FOR (which we set to 15) epochs have passed. If they have then we set model.embedding.weight.requires_grad to True, telling PyTorch that we should calculate gradients in the embedding layer and update them with our optimizer. [Refer](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/C%20-%20Loading%2C%20Saving%20and%20Freezing%20Embeddings.ipynb)

Model Architecture

    Seq2Seq(
      (encoder): Encoder(
        (tok_embedding): Embedding(2103, 256)
        (pos_embedding): Embedding(1000, 256)
        (layers): ModuleList(
          (0): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
          (1): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
          (2): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
        )
        (dropout): Dropout(p=0.15, inplace=False)
      )
      (decoder): Decoder(
        (tok_embedding): Embedding(5826, 256)
        (pos_embedding): Embedding(1000, 256)
        (layers): ModuleList(
          (0): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
          (1): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
          (2): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
          (3): DecoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (enc_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.15, inplace=False)
            )
            (dropout): Dropout(p=0.15, inplace=False)
          )
        )
        (fc_out): Linear(in_features=256, out_features=5826, bias=True)
        (dropout): Dropout(p=0.15, inplace=False)
      )
    )



Additional Parameters Used are:

Argument | Default Value |
---|---|
Batch Size | 28 |
Learning Rate | 0.0003|
Device | Cuda |
Loss Function | Cross Entropy Loss |
Test Loss | 1.523|


Training Log

    Epoch: 01 | Time: 0m 11s
    	Train Loss: 4.975 | Train PPL: 144.815
    	 Val. Loss: 3.446 |  Val. PPL:  31.370
    Epoch: 02 | Time: 0m 11s
    	Train Loss: 3.177 | Train PPL:  23.984
    	 Val. Loss: 2.794 |  Val. PPL:  16.352
    Epoch: 03 | Time: 0m 11s
    	Train Loss: 2.720 | Train PPL:  15.185
    	 Val. Loss: 2.482 |  Val. PPL:  11.964
    Epoch: 04 | Time: 0m 11s
    	Train Loss: 2.465 | Train PPL:  11.759
    	 Val. Loss: 2.326 |  Val. PPL:  10.234
    Epoch: 05 | Time: 0m 11s
    	Train Loss: 2.272 | Train PPL:   9.695
    	 Val. Loss: 2.187 |  Val. PPL:   8.906
    Epoch: 06 | Time: 0m 11s
    	Train Loss: 2.100 | Train PPL:   8.166
    	 Val. Loss: 2.067 |  Val. PPL:   7.899
    Epoch: 07 | Time: 0m 11s
    	Train Loss: 1.949 | Train PPL:   7.022
    	 Val. Loss: 1.968 |  Val. PPL:   7.159
    Epoch: 08 | Time: 0m 11s
    	Train Loss: 1.817 | Train PPL:   6.156
    	 Val. Loss: 1.885 |  Val. PPL:   6.584
    Epoch: 09 | Time: 0m 11s
    	Train Loss: 1.703 | Train PPL:   5.491
    	 Val. Loss: 1.814 |  Val. PPL:   6.137
    Epoch: 10 | Time: 0m 11s
    	Train Loss: 1.600 | Train PPL:   4.952
    	 Val. Loss: 1.769 |  Val. PPL:   5.865
    Epoch: 11 | Time: 0m 11s
    	Train Loss: 1.508 | Train PPL:   4.517
    	 Val. Loss: 1.697 |  Val. PPL:   5.456
    Epoch: 12 | Time: 0m 11s
    	Train Loss: 1.413 | Train PPL:   4.110
    	 Val. Loss: 1.652 |  Val. PPL:   5.217
    Epoch: 13 | Time: 0m 11s
    	Train Loss: 1.329 | Train PPL:   3.779
    	 Val. Loss: 1.609 |  Val. PPL:   4.996
    Epoch: 14 | Time: 0m 11s
    	Train Loss: 1.254 | Train PPL:   3.503
    	 Val. Loss: 1.574 |  Val. PPL:   4.828
    Epoch: 15 | Time: 0m 11s
    	Train Loss: 1.176 | Train PPL:   3.241
    	 Val. Loss: 1.533 |  Val. PPL:   4.633
    Epoch: 16 | Time: 0m 11s
    	Train Loss: 1.110 | Train PPL:   3.035
    	 Val. Loss: 1.508 |  Val. PPL:   4.516
    Epoch: 17 | Time: 0m 11s
    	Train Loss: 1.046 | Train PPL:   2.845
    	 Val. Loss: 1.479 |  Val. PPL:   4.390
    Epoch: 18 | Time: 0m 11s
    	Train Loss: 0.987 | Train PPL:   2.682
    	 Val. Loss: 1.463 |  Val. PPL:   4.319
    Epoch: 19 | Time: 0m 11s
    	Train Loss: 0.929 | Train PPL:   2.532
    	 Val. Loss: 1.433 |  Val. PPL:   4.190
    Epoch: 20 | Time: 0m 11s
    	Train Loss: 0.881 | Train PPL:   2.412
    	 Val. Loss: 1.429 |  Val. PPL:   4.176

Output 



    Enter the Description > write a program to add two numbers
    Bot:
      num1 = 1.5 
      num2 = 6.3 
      sum = num1 + num2 
      print ( f'Sum: {sum}' ) 


    Enter the Description > write a function to multiply two numbers
    Bot:
      def add_two_numbers ( num1 , num2 ) : 
         sum = num1 * num2 
         return sum 


    Enter the Description > write a function to find fibonacci number
    Bot:
      def fib ( n ) : 
         if n == 1 : 
             return n 
         else : 
             return n + recur_fibo ( n - 1 ) 


    Enter the Description > write a program to replace a string
    Bot:
      str1 = "Hello! It is a Good thing" 
      substr2 = "bad" 
      res = str1 . replace ( substr1 , substr2 ) 
      print ( "String after replace :" + str ( res ) ) 


    Enter the Description > write a program to find log of a number
    Bot:
      num = 10 
      print ( 'The square root of %0.3f is %0.3f' % ( num , num_sqrt ) ) 


    Enter the Description > write a function to find area of a circle
    Bot:
      def findArea ( r ) : 
         PI = 3.142 
         return PI * r * r 


    Enter the Description > write a program to reverse a list
    Bot:
      lst = [ 1 , 2 , 3 ] 
      print ( lst [ : : - 1 ] ) 


    Enter the Description > write a fucntion to find cosine angle
    Bot:

      def calcAngle ( hh , mm ) : 
         hour_angle = ( hh * 60 + mm ) 
         angle = abs ( hour_angle - minute_angle ) 
         angle = abs ( angle ) 
         return angle 
