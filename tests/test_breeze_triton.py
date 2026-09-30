"""Check fused sampling against the same top-50 categorical distribution."""

import pytest
import torch


@pytest.mark.parametrize("cfg", [False, True])
def test_triton_top50_matches_reference_cdf(cfg):
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from rho_tts.providers.breeze_triton import _sample_top50

    torch.manual_seed(527)
    half, vocab, codebook = 32, 2051, 2048
    batch = half * (2 if cfg else 1)
    logits = torch.randn(batch, 1, vocab, device="cuda", dtype=torch.float32)
    logits[..., codebook:] = 100.0
    temp = torch.rand(half, 1, device="cuda") + 0.5
    guidance = torch.rand(half, 1, device="cuda") * 3
    uniforms = torch.rand(half, device="cuda")
    output = torch.empty(batch, dtype=torch.long, device="cuda")

    _sample_top50[(half,)](logits, temp, guidance, uniforms, output,
                           vocab, codebook, half, cfg, 4096)

    scores = logits[:half, 0, :].clone()
    if cfg:
        scores = logits[half:, 0, :] + guidance * (scores - logits[half:, 0, :])
    scores[:, codebook:] = -float("inf")
    scores /= temp
    values, _ = torch.topk(scores, 50, dim=-1)
    scores = scores.masked_fill(scores < values[:, -1:], -float("inf"))
    weights = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    cumulative = weights.cumsum(dim=-1)
    expected = (cumulative <= uniforms[:, None] * weights.sum(dim=-1, keepdim=True)).sum(-1)
    assert torch.equal(output[:half], expected)
    if cfg:
        assert torch.equal(output[half:], expected)


def test_triton_flag_rejects_non_cuda_or_graph_modes():
    from rho_tts.providers.breeze import BreezeTTS

    for kwargs in ({"device": "cpu"}, {"fast": True}, {"cuda_graph_depth": True}):
        with pytest.raises(ValueError, match="eager CUDA"):
            BreezeTTS(instruction="Clear voice", triton_depth_sampling=True, **kwargs)
