"""Eager Breeze depth sampling for the default top-k, top-p=1 configuration.

The upstream eager sampler sorts the entire codec vocabulary after top-k even
when top-p=1 (so nucleus filtering cannot remove any candidates). This path
samples directly from the 50 retained logits. Other settings use upstream.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def _sample_topk_only(self, logits):
    if self.batch_size >= 2:
        cond = logits[: self.half, 0, :]
        uncond = logits[self.half :, 0, :]
        guided = uncond + self.guidance_scale * (cond - uncond)
    else:
        guided = logits[: self.half, 0, :]

    guided[..., self.codec_codebook_size : self.vocab_size] = float("-inf")
    scaled = guided / self.temperature_buf
    values, indices = torch.topk(scaled, 50, dim=-1)
    chosen = torch.multinomial(F.softmax(values, dim=-1), 1)
    tokens = indices.gather(-1, chosen).squeeze(-1)
    self._tok_buf[: self.half] = tokens
    if self.batch_size >= 2:
        self._tok_buf[self.half :] = tokens


def install_eager_topk_sampler(graph, *, use_triton=False):
    """Install a guarded sampler on an eager-only depth graph instance."""
    if not graph.no_graph or getattr(graph, "_rho_topk_installed", False):
        return
    if graph.codec_codebook_size < 50 or graph.codec_codebook_size > graph.vocab_size:
        return

    upstream_sample = graph._cfg_sample
    upstream_run = graph.run
    graph._rho_upstream_sample = upstream_sample
    graph._rho_upstream_run = upstream_run
    if use_triton:
        from .breeze_triton import sample_top50

        if graph.vocab_size > 4096:
            raise ValueError("Triton sampler supports vocabularies up to 4096")

        def sample(self, logits):
            sample_top50(self, logits)

        topk_sample = MethodType(sample, graph)
    else:
        topk_sample = MethodType(_sample_topk_only, graph)
    graph._rho_active_topk_sample = topk_sample
    previous_params = {}

    def run(backbone_hidden, first_cb_token, **params):
        if (
            params.get("top_k") == 50
            and params.get("top_p") == 1.0
            and params.get("do_sample") is True
        ):
            graph._cfg_sample = graph._rho_active_topk_sample
        else:
            graph._cfg_sample = upstream_sample
        # The buffers persist across eager frames. Refill only changed scalar
        # controls; each redundant fill otherwise launches a CUDA kernel.
        updates = {
            name: (
                None
                if (
                    isinstance(value, (bool, int, float))
                    and isinstance(previous_params.get(name), (bool, int, float))
                    and previous_params[name] == value
                )
                else value
            )
            for name, value in params.items()
        }
        result = upstream_run(backbone_hidden, first_cb_token, **updates)
        previous_params.update(params)
        return result

    graph.run = run
    graph._rho_topk_installed = True
    graph._rho_triton_installed = use_triton
