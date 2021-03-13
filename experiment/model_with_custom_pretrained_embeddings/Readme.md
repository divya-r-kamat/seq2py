# [Transformer Based Model + Pretrained Custom Embedding layer](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/model_with_custom_pretrained_embeddings)

## Pretrained Custom Embedding layer [(Notebook Link)](https://github.com/divya-r-kamat/seq2py/blob/main/experiment/model_with_custom_pretrained_embeddings/training_glove_word2vec_embedding.ipynb)

Trained a separate embedding layer for python keywords using glove so that model understands and pays special attention to whitespaces, colon and other things (like comma etc)
- Used CoNala dataset from [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github), to train the glove model.
- Faced some problem using the glove model directly into Pytorch model, so had ro convert glove model to word2vec using glove2word2vec library available in gensim
- Glove model is trained for 50 epochs and to generate embedding vectors of 256 dimension

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
