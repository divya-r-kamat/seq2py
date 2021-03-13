### [Transformer Based Model + Pretrained Custome Embedding + Data Augmentation](https://github.com/chinmay-singh/Propaganda/tree/lexicon)

## Pretrained Custom Embedding layer

Trained a separate embedding layer for python keywords using glove so that model understands and pays special attention to whitespaces, colon and other things (like comma etc)
- Used CoNala dataset from [CoNaLa: The Code/Natural Language Challenge](https://conala-corpus.github), to train the glove model.
- Faced some problem using the glove model directly into Pytorch model, so had ro convert glove model to word2vec using glove2word2vec library available in gensim
- Glove model is trained for 50 epochs and to generate embedding vectors of 256 dimension

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
Test Loss | 1.390


Training Log

    Epoch: 01 | Time: 0m 26s
        Train Loss: 3.672 | Train PPL:  39.313
         Val. Loss: 2.625 |  Val. PPL:  13.799
    Epoch: 02 | Time: 0m 27s
        Train Loss: 2.449 | Train PPL:  11.581
         Val. Loss: 2.229 |  Val. PPL:   9.291
    Epoch: 03 | Time: 0m 27s
        Train Loss: 2.083 | Train PPL:   8.032
         Val. Loss: 2.007 |  Val. PPL:   7.439
    Epoch: 04 | Time: 0m 27s
        Train Loss: 1.825 | Train PPL:   6.205
         Val. Loss: 1.851 |  Val. PPL:   6.367
    Epoch: 05 | Time: 0m 26s
        Train Loss: 1.616 | Train PPL:   5.031
         Val. Loss: 1.738 |  Val. PPL:   5.685
    Epoch: 06 | Time: 0m 27s
        Train Loss: 1.439 | Train PPL:   4.218
         Val. Loss: 1.658 |  Val. PPL:   5.247
    Epoch: 07 | Time: 0m 26s
        Train Loss: 1.289 | Train PPL:   3.628
         Val. Loss: 1.564 |  Val. PPL:   4.778
    Epoch: 08 | Time: 0m 27s
        Train Loss: 1.165 | Train PPL:   3.207
         Val. Loss: 1.515 |  Val. PPL:   4.550
    Epoch: 09 | Time: 0m 26s
        Train Loss: 1.060 | Train PPL:   2.886
         Val. Loss: 1.464 |  Val. PPL:   4.324
    Epoch: 10 | Time: 0m 26s
        Train Loss: 0.967 | Train PPL:   2.631
         Val. Loss: 1.449 |  Val. PPL:   4.259
    Epoch: 11 | Time: 0m 27s
        Train Loss: 0.861 | Train PPL:   2.365
         Val. Loss: 1.366 |  Val. PPL:   3.918
    Epoch: 12 | Time: 0m 27s
        Train Loss: 0.774 | Train PPL:   2.168
         Val. Loss: 1.328 |  Val. PPL:   3.775
    Epoch: 13 | Time: 0m 27s
        Train Loss: 0.705 | Train PPL:   2.024
         Val. Loss: 1.333 |  Val. PPL:   3.792
    Epoch: 14 | Time: 0m 27s
        Train Loss: 0.648 | Train PPL:   1.911
         Val. Loss: 1.317 |  Val. PPL:   3.734
    Epoch: 15 | Time: 0m 27s
        Train Loss: 0.596 | Train PPL:   1.814
         Val. Loss: 1.298 |  Val. PPL:   3.664
    Epoch: 16 | Time: 0m 26s
        Train Loss: 0.545 | Train PPL:   1.725
         Val. Loss: 1.290 |  Val. PPL:   3.633
    Epoch: 17 | Time: 0m 27s
        Train Loss: 0.505 | Train PPL:   1.657
         Val. Loss: 1.317 |  Val. PPL:   3.733
    Epoch: 18 | Time: 0m 27s
        Train Loss: 0.467 | Train PPL:   1.596
         Val. Loss: 1.296 |  Val. PPL:   3.653
    Epoch: 19 | Time: 0m 27s
        Train Loss: 0.436 | Train PPL:   1.547
         Val. Loss: 1.305 |  Val. PPL:   3.688
    Epoch: 20 | Time: 0m 26s
        Train Loss: 0.403 | Train PPL:   1.496
         Val. Loss: 1.323 |  Val. PPL:   3.755
         
 ## Highly overfit model
