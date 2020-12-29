gpt2_config = {
    'vocab_size_or_config_json_file': 50257,
    'n_ctx': 1024,
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'n_positions': 1024,
    'layer_norm_epsilon': 1e-5,
    'initializer_range': 0.02
}

distilgpt2_config = gpt2_config.copy()
distilgpt2_config.update({
    'n_layer': 6
})

gpt2_medium_config = gpt2_config.copy()
gpt2_medium_config.update({
    'n_embd': 1024,
    'n_head': 16,
    'n_layer': 24,
})

gpt2_large_config = gpt2_config.copy()
gpt2_large_config.update({
    'n_embd': 1280,
    'n_head': 20,
    'n_layer': 36
})

gpt2_xl_config = gpt2_config.copy()
gpt2_xl_config.update({
    'n_embd': 1600,
    'n_layer': 48,
    'n_head': 25
})

config = {
    'distilgpt2': (distilgpt2_config, 'distilgpt2-pytorch_model.bin'),
    'gpt2': (gpt2_config, 'gpt2-pytorch_model.bin'),
    'gpt2-medium': (gpt2_medium_config, 'gpt2-medium-pytorch_model.bin'),
    'gpt2-large': (gpt2_large_config, 'gpt2-large-pytorch_model.bin'),
    'gpt2-xl': (gpt2_xl_config, 'gpt2-xl-pytorch_model.bin')
}