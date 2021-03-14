# [Transformer Based Model + Pretrained Custom Embedding + Data Augmentation](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/model_with_augmentation_custom_pretrained_embeddings)

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
        (tok_embedding): Embedding(2141, 256)
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
        (tok_embedding): Embedding(6002, 256)
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
        (fc_out): Linear(in_features=256, out_features=6002, bias=True)
        (dropout): Dropout(p=0.15, inplace=False)
      )

Additional Parameters Used are:

Argument | Default Value |
---|---|
Batch Size | 28 |
Learning Rate | 0.0003|
Device | Cuda |
Loss Function | Cross Entropy Loss |
Test Loss | 1.443


Training Log

    Epoch: 01 | Time: 0m 28s
        Train Loss: 4.127 | Train PPL:  61.991
         Val. Loss: 2.814 |  Val. PPL:  16.673
    Epoch: 02 | Time: 0m 27s
        Train Loss: 2.646 | Train PPL:  14.103
         Val. Loss: 2.372 |  Val. PPL:  10.714
    Epoch: 03 | Time: 0m 27s
        Train Loss: 2.253 | Train PPL:   9.516
         Val. Loss: 2.135 |  Val. PPL:   8.454
    Epoch: 04 | Time: 0m 27s
        Train Loss: 1.988 | Train PPL:   7.303
         Val. Loss: 1.952 |  Val. PPL:   7.046
    Epoch: 05 | Time: 0m 27s
        Train Loss: 1.776 | Train PPL:   5.909
         Val. Loss: 1.836 |  Val. PPL:   6.271
    Epoch: 06 | Time: 0m 28s
        Train Loss: 1.575 | Train PPL:   4.830
         Val. Loss: 1.706 |  Val. PPL:   5.505
    Epoch: 07 | Time: 0m 28s
        Train Loss: 1.402 | Train PPL:   4.065
         Val. Loss: 1.606 |  Val. PPL:   4.984
    Epoch: 08 | Time: 0m 27s
        Train Loss: 1.259 | Train PPL:   3.520
         Val. Loss: 1.525 |  Val. PPL:   4.596
    Epoch: 09 | Time: 0m 28s
        Train Loss: 1.135 | Train PPL:   3.111
         Val. Loss: 1.457 |  Val. PPL:   4.294
    Epoch: 10 | Time: 0m 27s
        Train Loss: 1.021 | Train PPL:   2.777
         Val. Loss: 1.409 |  Val. PPL:   4.090
    Epoch: 11 | Time: 0m 27s
        Train Loss: 0.922 | Train PPL:   2.515
         Val. Loss: 1.358 |  Val. PPL:   3.889
    Epoch: 12 | Time: 0m 28s
        Train Loss: 0.836 | Train PPL:   2.306
         Val. Loss: 1.318 |  Val. PPL:   3.735
    Epoch: 13 | Time: 0m 27s
        Train Loss: 0.759 | Train PPL:   2.136
         Val. Loss: 1.289 |  Val. PPL:   3.630
    Epoch: 14 | Time: 0m 27s
        Train Loss: 0.692 | Train PPL:   1.997
         Val. Loss: 1.283 |  Val. PPL:   3.609
    Epoch: 15 | Time: 0m 28s
        Train Loss: 0.637 | Train PPL:   1.890
         Val. Loss: 1.265 |  Val. PPL:   3.545
    Epoch: 16 | Time: 0m 28s
        Train Loss: 0.583 | Train PPL:   1.791
         Val. Loss: 1.247 |  Val. PPL:   3.480
    Epoch: 17 | Time: 0m 28s
        Train Loss: 0.536 | Train PPL:   1.709
         Val. Loss: 1.241 |  Val. PPL:   3.458
    Epoch: 18 | Time: 0m 28s
        Train Loss: 0.497 | Train PPL:   1.644
         Val. Loss: 1.240 |  Val. PPL:   3.455
    Epoch: 19 | Time: 0m 27s
        Train Loss: 0.460 | Train PPL:   1.584
         Val. Loss: 1.246 |  Val. PPL:   3.477
    Epoch: 20 | Time: 0m 27s
        Train Loss: 0.431 | Train PPL:   1.539
         Val. Loss: 1.238 |  Val. PPL:   3.450         
