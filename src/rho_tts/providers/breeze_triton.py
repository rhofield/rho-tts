"""Fused Triton depth sampling for Breeze's top-k=50, top-p=1 eager path."""

import triton
import triton.language as tl


@triton.jit
def _sample_top50(
    Logits, Temperature, Guidance, Random, Output,
    VOCAB: tl.constexpr, CODEBOOK: tl.constexpr, HALF: tl.constexpr,
    CFG: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    cond = tl.load(Logits + row * VOCAB + index, index < CODEBOOK, other=-float("inf"))
    if CFG:
        uncond = tl.load(Logits + (row + HALF) * VOCAB + index,
                         index < CODEBOOK, other=-float("inf"))
        scale = tl.load(Guidance + row)
        scores = uncond + scale * (cond - uncond)
    else:
        scores = cond
    scores = tl.where(index < CODEBOOK, scores, -float("inf"))
    scores = scores / tl.load(Temperature + row)
    # The 50th value is the cutoff. Tied values at that boundary are all
    # retained, which is negligible for continuous logits but can differ
    # from torch.topk's tie selection.
    top = tl.topk(scores, 64)
    cutoff = tl.sum(tl.where(tl.arange(0, 64) == 49, top, 0.0), 0)
    scores = tl.where(scores >= cutoff, scores, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, 0))
    cumulative = tl.cumsum(weights, 0)
    choice = tl.min(tl.where(cumulative > tl.load(Random + row) * tl.sum(weights, 0),
                             index, BLOCK), 0)
    tl.store(Output + row, choice)
    if CFG:
        tl.store(Output + row + HALF, choice)


def sample_top50(graph, logits):
    """Write one sampled codec token per row into graph._tok_buf."""
    if not hasattr(graph, "_rho_triton_random") or graph._rho_triton_random.shape[0] != graph.half:
        import torch

        graph._rho_triton_random = torch.empty(graph.half, device=logits.device)
    graph._rho_triton_random.uniform_()
    _sample_top50[(graph.half,)](
        logits, graph.temperature_buf, graph.guidance_scale,
        graph._rho_triton_random, graph._tok_buf,
        graph.vocab_size, graph.codec_codebook_size, graph.half,
        graph.batch_size >= 2, triton.next_power_of_2(graph.vocab_size),
    )
