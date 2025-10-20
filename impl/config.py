from collections import namedtuple

GPTModelCfg = namedtuple(
    'ScratchModelConfig',
    [
        'vocab_size',
        'context_length',
        'emb_dim',
        'n_heads',
        'n_layers',
        'drop_rate',
        'qkv_bias',
    ]
)

GPT_2_CFG_124M = GPTModelCfg(
    vocab_size     = 50257,
    context_length = 1024,
    emb_dim        = 768,
    n_heads        = 12,
    n_layers       = 12,
    drop_rate      = 0.1,
    qkv_bias       = False
)

OPENAI_GPT_2_CFG_124M = GPTModelCfg(
    vocab_size     = 50257,
    context_length = 1024,
    emb_dim        = 768,
    n_heads        = 12,
    n_layers       = 12,
    drop_rate      = 0.1,
    qkv_bias       = True
)

OPENAI_GPT_2_CFG_355M = GPTModelCfg(
    vocab_size     = 50257,
    context_length = 1024,
    emb_dim        = 1024,
    n_heads        = 16,
    n_layers       = 24,
    drop_rate      = 0.0,
    qkv_bias       = True
)

def get_currently_chosen_cfg() -> GPTModelCfg:
    """ Returns the current model config to be used. Single point to change model configs """
    return GPT_2_CFG_124M