"""Run upstream EMScore with its documented CLIP token-feature extension.

See https://github.com/ShiYaya/EMScore at 20bce0ae234b3caeea3357c023c4d085c9a55bf0.
The adapter changes only this scorer's in-memory CLIP instance, caches its weights,
and leaves installed packages untouched.
"""
from types import MethodType, SimpleNamespace
from unittest.mock import patch


def _encode_text_with_tokens(model, text, local=False):
    """CLIP's text encoder with the local-token projection required by EMScore."""
    if not local:
        return model._dcc_original_encode_text(text)
    x = model.token_embedding(text).to(model.dtype)
    x = x + model.positional_embedding.to(model.dtype)
    x = model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    return model.ln_final(x).to(model.dtype) @ model.text_projection


def create_scorer(device):
    import clip
    import emscore.scorer as upstream

    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    model.eval()
    model._dcc_original_encode_text = model.encode_text
    model.encode_text = MethodType(_encode_text_with_tokens, model)
    backend = upstream.EMScorer(device=device)

    class VideoScorer:
        def score(self, **kwargs):
            # Upstream reloads CLIP on every score call. Reuse this instance and
            # substitute only upstream's module reference, restoring it on exit.
            proxy = SimpleNamespace(load=lambda *a, **k: (model, preprocess), tokenize=clip.tokenize)
            with patch.object(upstream, 'clip', proxy):
                return backend.score(**kwargs)

    return VideoScorer()
