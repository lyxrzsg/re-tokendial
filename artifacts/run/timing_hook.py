import json
import torch
from tokendial.inference import build_rgb_attention_mask_from_store

def generate(pipe, call, generation, scale, indices, folder, level):
    original = pipe.model_fn
    layers = [13,15,17,21]
    store = {}
    mask = None
    stages = []
    def wrapped(**kw):
        nonlocal mask
        if not kw['dit'].use_rgb_token:
            return original(**kw)
        step = len(stages) + 1
        if step == 2:
            kw.update(attn_collect_mode=True,attn_target_token_indices=indices,attn_collect_layers=layers,attn_store=store)
        if step >= 3:
            assert mask is not None
            kw['rgb_attention_mask'] = mask
        else:
            assert kw.get('rgb_attention_mask') is None
        prediction = original(**kw)
        if step == 2:
            assert {k[1] for k in store} == set(layers)
            mask = build_rgb_attention_mask_from_store(store, selected_layers=layers, quantile=.75).pow(.6)
            assert torch.isfinite(mask).all() and mask.min()<mask.max()
            torch.save(mask.detach().cpu(), folder/f'level{level}_mask.pt')
        stages.append('global' if step<=2 else 'masked')
        return prediction
    pipe.model_fn = wrapped
    try:
        result = call(**generation,tiled=True,rgb_cfg_scale=scale,rgb_attention_mask=None)
    finally:
        pipe.model_fn = original
    assert stages == ['global']*2+['masked']*48
    (folder/f'level{level}_trace.json').write_text(json.dumps(dict(stages=stages,indices=indices,keys=list(store))))
    return result
