# [Baseline Model](https://github.com/divya-r-kamat/seq2py/tree/main/experiment/baseline_model)

Model Architecture

    Seq2Seq(
      (encoder): Encoder(
        (tok_embedding): Embedding(2099, 256)
        (pos_embedding): Embedding(2000, 256)
        (layers): ModuleList(
          (0): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (dropout): Dropout(p=0.3, inplace=False)
          )
          (1): EncoderLayer(
            (self_attn_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (ff_layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (self_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (dropout): Dropout(p=0.3, inplace=False)
          )
        )
        (dropout): Dropout(p=0.3, inplace=False)
      )
      (decoder): Decoder(
        (tok_embedding): Embedding(6709, 256)
        (pos_embedding): Embedding(2000, 256)
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
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (dropout): Dropout(p=0.3, inplace=False)
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
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (encoder_attention): MultiHeadAttentionLayer(
              (fc_q): Linear(in_features=256, out_features=256, bias=True)
              (fc_k): Linear(in_features=256, out_features=256, bias=True)
              (fc_v): Linear(in_features=256, out_features=256, bias=True)
              (fc_o): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (positionwise_feedforward): PositionwiseFeedforwardLayer(
              (fc_1): Linear(in_features=256, out_features=512, bias=True)
              (fc_2): Linear(in_features=512, out_features=256, bias=True)
              (dropout): Dropout(p=0.3, inplace=False)
            )
            (dropout): Dropout(p=0.3, inplace=False)
          )
        )
        (fc_out): Linear(in_features=256, out_features=6709, bias=True)
        (dropout): Dropout(p=0.3, inplace=False)
      )
    )

Additional Parameters Used are:
Argument | Default Value |
---|---|
Batch Size | 16 |
Learning Rate | 0.0005|
Device | Cuda |
Loss Function | Cross Entropy Loss |
Test Loss | 1.725

Training Log

    Epoch: 01 | Time: 0m 12s
        Train Loss: 4.362 | Train PPL:  78.437
         Val. Loss: 3.137 |  Val. PPL:  23.029
    Epoch: 02 | Time: 0m 12s
        Train Loss: 2.906 | Train PPL:  18.282
         Val. Loss: 2.598 |  Val. PPL:  13.431
    Epoch: 03 | Time: 0m 12s
        Train Loss: 2.459 | Train PPL:  11.689
         Val. Loss: 2.350 |  Val. PPL:  10.483
    Epoch: 04 | Time: 0m 12s
        Train Loss: 2.174 | Train PPL:   8.792
         Val. Loss: 2.139 |  Val. PPL:   8.492
    Epoch: 05 | Time: 0m 12s
        Train Loss: 1.955 | Train PPL:   7.064
         Val. Loss: 2.002 |  Val. PPL:   7.401
    Epoch: 06 | Time: 0m 12s
        Train Loss: 1.781 | Train PPL:   5.934
         Val. Loss: 1.929 |  Val. PPL:   6.885
    Epoch: 07 | Time: 0m 12s
        Train Loss: 1.643 | Train PPL:   5.168
         Val. Loss: 1.859 |  Val. PPL:   6.419
    Epoch: 08 | Time: 0m 12s
        Train Loss: 1.525 | Train PPL:   4.596
         Val. Loss: 1.802 |  Val. PPL:   6.060
    Epoch: 09 | Time: 0m 12s
        Train Loss: 1.428 | Train PPL:   4.168
         Val. Loss: 1.744 |  Val. PPL:   5.722
    Epoch: 10 | Time: 0m 12s
        Train Loss: 1.347 | Train PPL:   3.844
         Val. Loss: 1.724 |  Val. PPL:   5.607

