MERaLiON2ForConditionalGeneration(
  (speech_encoder): WhisperEncoder(
    (conv1): Conv1d(128, 1280, kernel_size=(3,), stride=(1,), padding=(1,))
    (conv2): Conv1d(1280, 1280, kernel_size=(3,), stride=(2,), padding=(1,))
    (embed_positions): Embedding(1500, 1280)
    (layers): ModuleList(
      (0-31): 32 x WhisperEncoderLayer(
        (self_attn): WhisperAttention(
          (k_proj): Linear(in_features=1280, out_features=1280, bias=False)
          (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (activation_fn): GELUActivation()
        (fc1): Linear(in_features=1280, out_features=5120, bias=True)
        (fc2): Linear(in_features=5120, out_features=1280, bias=True)
        (final_layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      )
    )
    (layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
  )
  (ln_speech): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
  (speech_audio_adapter): MERaLiON2SpeechAudioAdaperLarge(
    (mlp_adapter): Sequential(
      (0): Linear(in_features=19200, out_features=6400, bias=True)
      (1): SiLU()
      (2): Dropout(p=0.01, inplace=False)
    )
    (gate_proj): Linear(in_features=6400, out_features=6400, bias=True)
    (pool_proj): Linear(in_features=6400, out_features=6400, bias=True)
    (act_fn): SiLU()
    (out_proj): Linear(in_features=6400, out_features=3584, bias=True)
  )
  (text_decoder): Gemma2ForCausalLM(
    (model): Gemma2Model(
      (embed_tokens): Embedding(256000, 3584, padding_idx=0)
      (layers): ModuleList(
        (0-41): 42 x Gemma2DecoderLayer(
          (self_attn): Gemma2Attention(
            (q_proj): Linear(in_features=3584, out_features=4096, bias=False)
            (k_proj): Linear(in_features=3584, out_features=2048, bias=False)
            (v_proj): Linear(in_features=3584, out_features=2048, bias=False)
            (o_proj): Linear(in_features=4096, out_features=3584, bias=False)
          )
          (mlp): Gemma2MLP(
            (gate_proj): Linear(in_features=3584, out_features=14336, bias=False)
            (up_proj): Linear(in_features=3584, out_features=14336, bias=False)
            (down_proj): Linear(in_features=14336, out_features=3584, bias=False)
            (act_fn): GELUTanh()
          )
          (input_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
          (post_attention_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
          (pre_feedforward_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
          (post_feedforward_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        )
      )
      (norm): Gemma2RMSNorm((3584,), eps=1e-06)
      (rotary_emb): Gemma2RotaryEmbedding()
    )
    (lm_head): Linear(in_features=3584, out_features=256000, bias=False)
  )
)