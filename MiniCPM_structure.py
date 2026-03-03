# MiniCPM-2B-128k
MiniCPMForCausalLM(
  (model): MiniCPMModel(
    (embed_tokens): Embedding(122760, 2304)
    (layers): ModuleList(
      (0-39): 40 x MiniCPMDecoderLayer(
        (self_attn): MiniCPMSdpaAttention(
          (q_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (rotary_emb): MiniCPMDynamicNTKScalingRotaryEmbedding()
        )
        (mlp): MiniCPMMLP(
          (gate_proj): Linear(in_features=2304, out_features=5760, bias=False)
          (up_proj): Linear(in_features=2304, out_features=5760, bias=False)
          (down_proj): Linear(in_features=5760, out_features=2304, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MiniCPMRMSNorm()
        (post_attention_layernorm): MiniCPMRMSNorm()
      )
    )
    (norm): MiniCPMRMSNorm()
  )
  (lm_head): Linear(in_features=2304, out_features=122760, bias=False)
)

# MiniCPM-2B-128k-pruned-ch-0.3-cuda
