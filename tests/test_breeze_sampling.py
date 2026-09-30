"""The eager sampling shortcut must preserve top-k behavior and fall back safely."""

import torch

from rho_tts.providers.breeze_sampling import _sample_topk_only, install_eager_topk_sampler


class FakeDepthGraph:
    no_graph = True
    batch_size = 1
    half = 1
    codec_codebook_size = 64
    vocab_size = 66

    def __init__(self):
        self.temperature_buf = torch.ones(1, 1)
        self.guidance_scale = torch.ones(1, 1)
        self._tok_buf = torch.zeros(1, dtype=torch.long)
        self.upstream_sample = self._cfg_sample
        self.calls = []

    def _cfg_sample(self, logits):
        self.calls.append("upstream")

    def run(self, _hidden, _token, **params):
        self.calls.append((self._cfg_sample.__func__ is _sample_topk_only, params))


def test_shortcut_uses_only_valid_topk_tokens():
    graph = FakeDepthGraph()
    logits = torch.full((1, 1, 66), -100.0)
    logits[0, 0, 7] = 10.0
    logits[0, 0, 64:] = 100.0  # reserved tokens must never be sampled
    _sample_topk_only(graph, logits)
    assert graph._tok_buf.item() == 7


def test_shortcut_dispatch_and_unchanged_controls():
    graph = FakeDepthGraph()
    install_eager_topk_sampler(graph)
    defaults = dict(top_k=50, top_p=1.0, do_sample=True, temperature=0.9)
    graph.run(None, None, **defaults)
    graph.run(None, None, **defaults)
    graph.run(None, None, **{**defaults, "top_p": 0.9})
    assert graph.calls[0] == (True, defaults)
    assert graph.calls[1] == (True, {key: None for key in defaults})
    assert graph.calls[2][0] is False
    assert graph.calls[2][1]["top_p"] == 0.9
