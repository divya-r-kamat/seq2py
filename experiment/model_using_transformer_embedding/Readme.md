# [Transformer Based model(Learned Embeddings)](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/model_using_transformer_embedding)

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
Test Loss | 1.410

Training Log

    Epoch: 01 | Time: 0m 14s
    	Train Loss: 4.782 | Train PPL: 119.377
    	 Val. Loss: 3.156 |  Val. PPL:  23.485
    Epoch: 02 | Time: 0m 14s
    	Train Loss: 2.948 | Train PPL:  19.068
    	 Val. Loss: 2.637 |  Val. PPL:  13.965
    Epoch: 03 | Time: 0m 14s
    	Train Loss: 2.534 | Train PPL:  12.606
    	 Val. Loss: 2.364 |  Val. PPL:  10.638
    Epoch: 04 | Time: 0m 14s
    	Train Loss: 2.284 | Train PPL:   9.819
    	 Val. Loss: 2.199 |  Val. PPL:   9.018
    Epoch: 05 | Time: 0m 14s
    	Train Loss: 2.093 | Train PPL:   8.110
    	 Val. Loss: 2.051 |  Val. PPL:   7.773
    Epoch: 06 | Time: 0m 14s
    	Train Loss: 1.931 | Train PPL:   6.898
    	 Val. Loss: 1.957 |  Val. PPL:   7.077
    Epoch: 07 | Time: 0m 14s
    	Train Loss: 1.795 | Train PPL:   6.022
    	 Val. Loss: 1.875 |  Val. PPL:   6.519
    Epoch: 08 | Time: 0m 14s
    	Train Loss: 1.674 | Train PPL:   5.333
    	 Val. Loss: 1.796 |  Val. PPL:   6.025
    Epoch: 09 | Time: 0m 14s
    	Train Loss: 1.561 | Train PPL:   4.763
    	 Val. Loss: 1.728 |  Val. PPL:   5.630
    Epoch: 10 | Time: 0m 15s
    	Train Loss: 1.458 | Train PPL:   4.298
    	 Val. Loss: 1.690 |  Val. PPL:   5.419
    Epoch: 11 | Time: 0m 15s
    	Train Loss: 1.363 | Train PPL:   3.908
    	 Val. Loss: 1.626 |  Val. PPL:   5.084
    Epoch: 12 | Time: 0m 14s
    	Train Loss: 1.273 | Train PPL:   3.571
    	 Val. Loss: 1.581 |  Val. PPL:   4.860
    Epoch: 13 | Time: 0m 15s
    	Train Loss: 1.197 | Train PPL:   3.310
    	 Val. Loss: 1.550 |  Val. PPL:   4.709
    Epoch: 14 | Time: 0m 15s
    	Train Loss: 1.124 | Train PPL:   3.077
    	 Val. Loss: 1.498 |  Val. PPL:   4.474
    Epoch: 15 | Time: 0m 15s
    	Train Loss: 1.047 | Train PPL:   2.849
    	 Val. Loss: 1.470 |  Val. PPL:   4.349
    Epoch: 16 | Time: 0m 15s
    	Train Loss: 0.985 | Train PPL:   2.677
    	 Val. Loss: 1.446 |  Val. PPL:   4.246
    Epoch: 17 | Time: 0m 15s
    	Train Loss: 0.933 | Train PPL:   2.543
    	 Val. Loss: 1.428 |  Val. PPL:   4.172
    Epoch: 18 | Time: 0m 15s
    	Train Loss: 0.873 | Train PPL:   2.393
    	 Val. Loss: 1.405 |  Val. PPL:   4.076
    Epoch: 19 | Time: 0m 15s
    	Train Loss: 0.824 | Train PPL:   2.281
    	 Val. Loss: 1.385 |  Val. PPL:   3.995
    Epoch: 20 | Time: 0m 15s
    	Train Loss: 0.780 | Train PPL:   2.182
    	 Val. Loss: 1.363 |  Val. PPL:   3.908
