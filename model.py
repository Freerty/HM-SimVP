"""HM-SimVP: local SimVPv2/gSTA backbone, RDS encoder, edge-AHG and MCRP.

Extracted from the existing project implementation. Public constructor: HMSimVP.
Input [B,12,H,W], output [B,6,H,W]; return_aux=True also returns training terms.
The backbone follows the original SimVPv2/gSTA implementation (OpenSTL-style).
Requires torch only. No OpenSTL runtime or other project files are imported.
"""
from __future__ import annotations
import math
LOCAL_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
from collections.abc import Sequence
import torch
from torch import nn
import torch.nn.functional as F
def _sampling_generator(num_layers: int, reverse: bool = False) -> list[bool]:
    samplings = [False, True] * (num_layers // 2)
    samplings = samplings[:num_layers]
    return list(reversed(samplings)) if reverse else samplings


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        upsampling: bool = False,
        act_norm: bool = False,
        act_inplace: bool = True,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        norm_groups = 2 if out_channels % 2 == 0 else 1
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
        act_norm: bool = True,
        act_inplace: bool = True,
    ) -> None:
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            upsampling=upsampling,
            padding=padding,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, c_in: int, c_hid: int, n_s: int, spatio_kernel: int) -> None:
        super().__init__()
        samplings = _sampling_generator(n_s)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, spatio_kernel, downsampling=samplings[0], act_inplace=False),
            *[
                ConvSC(c_hid, c_hid, spatio_kernel, downsampling=sampling, act_inplace=False)
                for sampling in samplings[1:]
            ],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.enc[0](x)
        latent = enc1
        for layer in self.enc[1:]:
            latent = layer(latent)
        return latent, enc1


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 21, dilation: int = 3) -> None:
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation
        )
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.activation(self.proj_1(x))
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        f_g = self.conv1(attn)
        f_x, g_x = torch.chunk(f_g, 2, dim=1)
        x = torch.sigmoid(g_x) * f_x
        return self.proj_2(x) + shortcut


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class MixMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class GASubBlock(nn.Module):
    """gSTA block used by SimVPv2."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 21,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        init_value: float = 1e-2,
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MixMlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(
            self.layer_scale_1.view(1, -1, 1, 1) * self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.view(1, -1, 1, 1) * self.mlp(self.norm2(x))
        )
        return x


class OriginalGSTATranslatorBlock(nn.Module):
    """OpenSTL-style MetaBlock: gSTA first, optional 1x1 channel reduction after."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        kernel_size: int = 21,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.block = GASubBlock(
            self.in_channels,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )
        self.reduction = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(self.block(x))


class OriginalMidMetaNet(nn.Module):
    """OpenSTL SimVPv2/gSTA hidden Translator."""

    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        n_t: int,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        kernel_size: int = 21,
    ) -> None:
        super().__init__()
        if n_t < 2:
            raise ValueError("N_T must be at least 2 for SimVPv2.")
        if mlp_ratio <= 1:
            raise ValueError("mlp_ratio must be greater than 1.")

        drop_rates = torch.linspace(1e-2, drop_path, n_t).tolist()
        layers: list[nn.Module] = [
            OriginalGSTATranslatorBlock(
                channel_in,
                channel_hid,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_rates[0],
                kernel_size=kernel_size,
            )
        ]
        for index in range(1, n_t - 1):
            layers.append(
                OriginalGSTATranslatorBlock(
                    channel_hid,
                    channel_hid,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_rates[index],
                    kernel_size=kernel_size,
                )
            )
        layers.append(
            OriginalGSTATranslatorBlock(
                channel_hid,
                channel_in,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                kernel_size=kernel_size,
            )
        )
        self.enc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        z = x.reshape(batch_size, time_steps * channels, height, width)
        z = self.enc(z)
        return z.reshape(batch_size, time_steps, channels, height, width)


class AdaptiveHypergraphLatentMixer(nn.Module):
    """Content-adaptive hypergraph message passing for image latent tokens."""

    def __init__(
        self,
        channels: int,
        embed_channels: int = 16,
        top_k: int = 9,
        dropout: float = 0.05,
        residual_scale: float = 0.35,
        gate_bias: float = -0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.embed_channels = int(embed_channels)
        self.top_k = int(top_k)

        self.embed = nn.Sequential(
            nn.Conv2d(self.channels, self.embed_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.embed_channels),
            nn.GELU(),
        )
        self.message_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _build_hyperedge_weights(self, context: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = context.shape
        num_nodes = height * width
        k = min(max(self.top_k, 1), num_nodes)

        emb = self.embed(context).flatten(2).transpose(1, 2).contiguous()
        emb = F.normalize(emb.float(), dim=-1, eps=1e-6)
        similarity = torch.bmm(emb, emb.transpose(1, 2))
        values, indices = torch.topk(similarity, k=k, dim=-1)
        weights = torch.softmax(values, dim=-1).to(context.dtype)

        hyperedges = context.new_zeros(batch_size, num_nodes, num_nodes)
        hyperedges.scatter_(dim=-1, index=indices, src=weights)
        return hyperedges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"AdaptiveHypergraphLatentMixer expects [B,T,C,H,W], got {tuple(x.shape)}")

        batch_size, time_steps, channels, height, width = x.shape
        context = x.mean(dim=1)
        hyperedges = self._build_hyperedge_weights(context)

        tokens = x.permute(0, 1, 3, 4, 2).reshape(batch_size, time_steps, height * width, channels)
        edge_tokens = torch.einsum("ben,btnc->btec", hyperedges, tokens)
        node_tokens = torch.einsum("ben,btec->btnc", hyperedges, edge_tokens)
        node_norm = hyperedges.sum(dim=1).clamp_min(1e-6).view(batch_size, 1, height * width, 1)
        node_tokens = node_tokens / node_norm

        message = node_tokens.reshape(batch_size, time_steps, height, width, channels)
        message = message.permute(0, 1, 4, 2, 3).reshape(batch_size * time_steps, channels, height, width)
        message = self.message_proj(message)
        base = x.reshape(batch_size * time_steps, channels, height, width)
        gate = torch.sigmoid(self.gate(torch.cat([base, message], dim=1)))
        mixed = base + self.residual_scale.to(base.dtype) * gate * message
        return mixed.reshape(batch_size, time_steps, channels, height, width)


class LocalOverlapHyperedgeInteractionMixer(AdaptiveHypergraphLatentMixer):
    """Insert incidence-compatible local edge interaction into content AHG."""

    def __init__(
        self,
        channels: int,
        embed_channels: int = 16,
        top_k: int = 7,
        dropout: float = 0.05,
        residual_scale: float = 0.18,
        gate_bias: float = -0.85,
        overlap_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            channels=channels,
            embed_channels=embed_channels,
            top_k=top_k,
            dropout=dropout,
            residual_scale=residual_scale,
            gate_bias=gate_bias,
        )
        if overlap_eps < 0.0:
            raise ValueError("overlap_eps must be non-negative.")
        self.overlap_eps = float(overlap_eps)

        # The constructor of Linear normally consumes RNG even though its
        # formal initialization is zero.  Isolate it so every shared parameter
        # keeps the same seeded initialization as the established AHG model.
        with torch.random.fork_rng(devices=[]):
            self.edge_adapter = nn.Linear(
                self.channels,
                self.channels,
                bias=False,
            )
        nn.init.zeros_(self.edge_adapter.weight)

        self.edge_interaction_enabled = True
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _build_content_hypergraph(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dense incidence and the exact sparse Top-k representation."""
        batch_size, _, height, width = context.shape
        num_nodes = height * width
        k = min(max(self.top_k, 1), num_nodes)

        embedding = self.embed(context).flatten(2).transpose(1, 2).contiguous()
        embedding = F.normalize(embedding.float(), dim=-1, eps=1e-6)
        similarity = torch.bmm(embedding, embedding.transpose(1, 2))
        values, indices = torch.topk(similarity, k=k, dim=-1)
        weights = torch.softmax(values, dim=-1).to(context.dtype)

        hyperedges = context.new_zeros(batch_size, num_nodes, num_nodes)
        hyperedges.scatter_(dim=-1, index=indices, src=weights)
        return hyperedges, indices, weights

    @staticmethod
    def _local_neighbours(
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return ``[N,8]`` non-self neighbours with -1 at boundaries."""
        grid = torch.arange(height * width, device=device).reshape(height, width)
        neighbours = torch.full(
            (height * width, len(LOCAL_OFFSETS)),
            -1,
            device=device,
            dtype=torch.long,
        )
        for slot, (dy, dx) in enumerate(LOCAL_OFFSETS):
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            source = grid[source_y, source_x].reshape(-1)
            target = grid[target_y, target_x].reshape(-1)
            neighbours[source, slot] = target
        return neighbours

    @staticmethod
    def _weighted_local_overlap(
        indices: torch.Tensor,
        weights: torch.Tensor,
        neighbours: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted Top-k intersections for local non-self pairs."""
        batch_size, num_edges, _ = indices.shape
        overlaps = weights.new_zeros(
            batch_size,
            num_edges,
            neighbours.shape[1],
        )
        for slot in range(neighbours.shape[1]):
            neighbour = neighbours[:, slot]
            valid = neighbour >= 0
            if not torch.any(valid):
                continue
            center_indices = indices[:, valid]
            center_weights = weights[:, valid]
            other_indices = indices[:, neighbour[valid]]
            other_weights = weights[:, neighbour[valid]]
            matches = center_indices.unsqueeze(-1) == other_indices.unsqueeze(-2)
            pair_weights = torch.minimum(
                center_weights.unsqueeze(-1),
                other_weights.unsqueeze(-2),
            )
            overlap_values = (
                matches.to(pair_weights.dtype) * pair_weights
            ).sum(dim=(-1, -2))
            overlaps[:, valid, slot] = overlap_values.to(overlaps.dtype)
        return overlaps

    def _interact_hyperedges(
        self,
        edge_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply sparse local edge-to-edge interaction and return diagnostics."""
        neighbours = self._local_neighbours(height, width, edge_tokens.device)
        overlaps = self._weighted_local_overlap(indices, weights, neighbours)
        valid_overlap = overlaps > self.overlap_eps
        overlap_mass = torch.where(valid_overlap, overlaps, 0.0).sum(dim=-1)
        attention = torch.where(
            valid_overlap,
            overlaps / overlap_mass.clamp_min(1e-6).unsqueeze(-1),
            0.0,
        )

        neighbour_tokens = torch.zeros_like(edge_tokens)
        for slot in range(neighbours.shape[1]):
            neighbour = neighbours[:, slot]
            valid_spatial = neighbour >= 0
            if not torch.any(valid_spatial):
                continue
            neighbour_tokens[:, :, valid_spatial] += (
                attention[:, valid_spatial, slot][:, None, :, None]
                * edge_tokens[:, :, neighbour[valid_spatial]]
            )

        has_valid = valid_overlap.any(dim=-1)
        neighbour_tokens = torch.where(
            has_valid[:, None, :, None],
            neighbour_tokens,
            edge_tokens,
        )
        edge_residual = neighbour_tokens - edge_tokens
        if self.edge_interaction_enabled:
            edge_delta = self.edge_adapter(edge_residual)
            edge_delta = edge_delta * has_valid[:, None, :, None].to(
                edge_delta.dtype
            )
        else:
            edge_delta = torch.zeros_like(edge_residual)
        refined = edge_tokens + edge_delta

        positive_count = valid_overlap.sum().clamp_min(1)
        positive_values = torch.where(valid_overlap, overlaps, 0.0)
        positive_mean = positive_values.sum() / positive_count
        positive_second = positive_values.square().sum() / positive_count
        positive_std = (positive_second - positive_mean.square()).clamp_min(0).sqrt()
        edge_rms = edge_tokens.float().square().mean().sqrt()
        residual_rms = edge_residual.float().square().mean().sqrt()
        delta_rms = edge_delta.float().square().mean().sqrt()
        diagnostics = {
            "positive_overlap_fraction": has_valid.float().mean().detach(),
            "positive_neighbours_mean": valid_overlap.float().sum(-1).mean().detach(),
            "no_positive_neighbour_fraction": (~has_valid).float().mean().detach(),
            "positive_overlap_mean": positive_mean.float().detach(),
            "positive_overlap_std": positive_std.float().detach(),
            "edge_token_rms": edge_rms.detach(),
            "edge_residual_rms": residual_rms.detach(),
            "edge_residual_ratio": (residual_rms / edge_rms.clamp_min(1e-6)).detach(),
            "edge_delta_rms": delta_rms.detach(),
            "edge_delta_ratio": (delta_rms / edge_rms.clamp_min(1e-6)).detach(),
            "edge_interaction_enabled": edge_tokens.new_tensor(
                float(self.edge_interaction_enabled)
            ).detach(),
        }
        return refined, diagnostics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "LocalOverlapHyperedgeInteractionMixer expects "
                f"[B,T,C,H,W], got {tuple(x.shape)}"
            )
        batch_size, time_steps, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        num_nodes = height * width
        context = x.mean(dim=1)
        hyperedges, indices, weights = self._build_content_hypergraph(context)

        tokens = x.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            time_steps,
            num_nodes,
            channels,
        )
        edge_tokens = torch.einsum("ben,btnc->btec", hyperedges, tokens)
        refined_edges, diagnostics = self._interact_hyperedges(
            edge_tokens,
            indices,
            weights,
            height,
            width,
        )
        node_tokens = torch.einsum(
            "ben,btec->btnc",
            hyperedges,
            refined_edges,
        )
        node_norm = hyperedges.sum(dim=1).clamp_min(1e-6).view(
            batch_size,
            1,
            num_nodes,
            1,
        )
        node_tokens = node_tokens / node_norm

        message = node_tokens.reshape(
            batch_size,
            time_steps,
            height,
            width,
            channels,
        ).permute(0, 1, 4, 2, 3)
        message = self.message_proj(
            message.reshape(batch_size * time_steps, channels, height, width)
        )
        base = x.reshape(batch_size * time_steps, channels, height, width)
        gate = torch.sigmoid(self.gate(torch.cat([base, message], dim=1)))
        mixed = base + self.residual_scale.to(base.dtype) * gate * message
        self.last_diagnostics = diagnostics
        return mixed.reshape(
            batch_size,
            time_steps,
            channels,
            height,
            width,
        )


class Decoder(nn.Module):
    def __init__(self, c_hid: int, c_out: int, n_s: int, spatio_kernel: int) -> None:
        super().__init__()
        samplings = _sampling_generator(n_s, reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvSC(c_hid, c_hid, spatio_kernel, upsampling=sampling, act_inplace=False)
                for sampling in samplings
            ]
        )
        self.readout = nn.Conv2d(c_hid, c_out, 1)

    def forward(self, hidden: torch.Tensor, enc1: torch.Tensor) -> torch.Tensor:
        for layer in self.dec[:-1]:
            hidden = layer(hidden)
        y = self.dec[-1](hidden + enc1)
        return self.readout(y)


class SimVP2OriginalEdgeInteractionAHGCore(nn.Module):
    """Original SimVPv2 core with edge-interaction AHG after gSTA."""

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 2,
        N_T: int = 8,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        gsta_kernel_size: int = 21,
        ahg_embed_channels: int = 16,
        ahg_top_k: int = 7,
        ahg_dropout: float = 0.05,
        ahg_residual_scale: float = 0.18,
        ahg_gate_bias: float = -0.85,
        overlap_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        input_length, channels, _, _ = in_shape
        self.enc = Encoder(channels, hid_S, N_S, spatio_kernel_enc)
        self.hid = OriginalMidMetaNet(
            input_length * hid_S,
            hid_T,
            N_T,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            kernel_size=gsta_kernel_size,
        )
        self.ahg = LocalOverlapHyperedgeInteractionMixer(
            channels=hid_S,
            embed_channels=ahg_embed_channels,
            top_k=ahg_top_k,
            dropout=ahg_dropout,
            residual_scale=ahg_residual_scale,
            gate_bias=ahg_gate_bias,
            overlap_eps=overlap_eps,
        )
        self.dec = Decoder(hid_S, channels, N_S, spatio_kernel_dec)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x_raw.shape
        encoded, skip = self.enc(
            x_raw.reshape(batch_size * time_steps, channels, height, width)
        )
        _, hidden_channels, hidden_height, hidden_width = encoded.shape
        latent = encoded.reshape(
            batch_size,
            time_steps,
            hidden_channels,
            hidden_height,
            hidden_width,
        )
        hidden = self.ahg(self.hid(latent)).reshape(
            batch_size * time_steps,
            hidden_channels,
            hidden_height,
            hidden_width,
        )
        decoded = self.dec(hidden, skip)
        return decoded.reshape(
            batch_size,
            time_steps,
            channels,
            height,
            width,
        )


class RMSNorm2d(nn.Module):
    """Per-pixel RMS normalization over the channel dimension."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"RMSNorm2d expects [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        inverse_rms = torch.rsqrt(
            x.float().square().mean(dim=1, keepdim=True) + self.eps
        ).to(x.dtype)
        weight = self.weight.to(x.dtype).view(1, self.channels, 1, 1)
        return x * inverse_rms * weight


class DualScaleGatedEncoderBlock(nn.Module):
    """RMS-pre-norm residual interaction of local and dilated spatial cues."""

    def __init__(
        self,
        channels: int,
        expansion_ratio: int = 1,
        context_dilation: int = 2,
        rms_eps: float = 1e-6,
        beta_init: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive.")
        if context_dilation <= 0:
            raise ValueError("context_dilation must be positive.")

        self.channels = int(channels)
        self.expanded_channels = self.channels * int(expansion_ratio)
        self.context_dilation = int(context_dilation)
        self.norm = RMSNorm2d(self.channels, eps=rms_eps)
        self.expand = nn.Conv2d(
            self.channels,
            2 * self.expanded_channels,
            kernel_size=1,
        )
        self.local_dw = nn.Conv2d(
            self.expanded_channels,
            self.expanded_channels,
            kernel_size=3,
            padding=1,
            groups=self.expanded_channels,
        )
        self.context_dw = nn.Conv2d(
            self.expanded_channels,
            self.expanded_channels,
            kernel_size=3,
            padding=self.context_dilation,
            dilation=self.context_dilation,
            groups=self.expanded_channels,
        )
        self.project = nn.Conv2d(
            self.expanded_channels,
            self.channels,
            kernel_size=1,
        )
        self.beta = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(beta_init))
        )
        self.residual_enabled = True
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(self.norm(x))
        local, context = expanded.chunk(2, dim=1)
        local = self.local_dw(local)
        context = self.context_dw(context)
        branch = self.project(F.silu(local) * context)
        applied = self.beta.to(branch.dtype) * branch
        if not self.residual_enabled:
            applied = torch.zeros_like(applied)

        base_rms = x.detach().float().square().mean().sqrt()
        branch_rms = branch.detach().float().square().mean().sqrt()
        applied_rms = applied.detach().float().square().mean().sqrt()
        self.last_diagnostics = {
            "beta_abs_mean": self.beta.detach().float().abs().mean(),
            "branch_rms": branch_rms,
            "applied_update_rms": applied_rms,
            "applied_update_ratio": applied_rms / base_rms.clamp_min(1e-6),
            "residual_enabled": x.new_tensor(float(self.residual_enabled)).detach(),
        }
        return x + applied


class SimVP2OriginalRMSDSEdgeAHGCore(SimVP2OriginalEdgeInteractionAHGCore):
    """Established edge-AHG core with two encoder-only refinement blocks."""

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 2,
        N_T: int = 8,
        mlp_ratio: float = 8.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        gsta_kernel_size: int = 21,
        ahg_embed_channels: int = 16,
        ahg_top_k: int = 7,
        ahg_dropout: float = 0.05,
        ahg_residual_scale: float = 0.18,
        ahg_gate_bias: float = -0.85,
        overlap_eps: float = 1e-6,
        encoder_expansion_ratio: int = 1,
        encoder_context_dilation: int = 2,
        encoder_rms_eps: float = 1e-6,
        encoder_beta_init: float = 0.0,
    ) -> None:
        if N_S != 2:
            raise ValueError(
                "The RMS dual-scale encoder experiment is defined for N_S=2."
            )
        super().__init__(
            in_shape=in_shape,
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            spatio_kernel_enc=spatio_kernel_enc,
            spatio_kernel_dec=spatio_kernel_dec,
            gsta_kernel_size=gsta_kernel_size,
            ahg_embed_channels=ahg_embed_channels,
            ahg_top_k=ahg_top_k,
            ahg_dropout=ahg_dropout,
            ahg_residual_scale=ahg_residual_scale,
            ahg_gate_bias=ahg_gate_bias,
            overlap_eps=overlap_eps,
        )

        # These extra modules must not advance the shared initialization RNG.
        # This keeps every established parameter seed-matched to the baseline.
        with torch.random.fork_rng(devices=[]):
            self.encoder_refine_full = DualScaleGatedEncoderBlock(
                hid_S,
                expansion_ratio=encoder_expansion_ratio,
                context_dilation=encoder_context_dilation,
                rms_eps=encoder_rms_eps,
                beta_init=encoder_beta_init,
            )
            self.encoder_refine_latent = DualScaleGatedEncoderBlock(
                hid_S,
                expansion_ratio=encoder_expansion_ratio,
                context_dilation=encoder_context_dilation,
                rms_eps=encoder_rms_eps,
                beta_init=encoder_beta_init,
            )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x_raw.shape
        frames = x_raw.reshape(batch_size * time_steps, channels, height, width)
        skip = self.encoder_refine_full(self.enc.enc[0](frames))
        encoded = self.encoder_refine_latent(self.enc.enc[1](skip))
        _, hidden_channels, hidden_height, hidden_width = encoded.shape
        latent = encoded.reshape(
            batch_size,
            time_steps,
            hidden_channels,
            hidden_height,
            hidden_width,
        )
        hidden = self.ahg(self.hid(latent)).reshape(
            batch_size * time_steps,
            hidden_channels,
            hidden_height,
            hidden_width,
        )
        decoded = self.dec(hidden, skip)
        return decoded.reshape(
            batch_size,
            time_steps,
            channels,
            height,
            width,
        )

    def encoder_diagnostics(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "full_resolution": self.encoder_refine_full.last_diagnostics,
            "latent_resolution": self.encoder_refine_latent.last_diagnostics,
        }


class SignedResidualCorrection(nn.Module):
    """The performance-critical signed residual path of the old strong model."""

    def __init__(self, input_length: int, output_length: int, hidden: int = 32, scale: float = 0.25):
        super().__init__()
        self.scale = float(scale)
        self.trunk = nn.Sequential(
            nn.Conv2d(input_length + output_length, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.residual_head = nn.Conv2d(hidden, output_length, 3, padding=1)

    def forward(self, history: torch.Tensor, generated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(torch.cat((history, generated), dim=1))
        # Match the empirically strong signed residual path exactly.  Bounding
        # is handled by normalized training targets, not by a tanh bottleneck.
        residual = self.scale * self.residual_head(features)
        return generated + residual, residual


class GatedMomentConstrainedResidualProjection(nn.Module):
    """Moment projection whose increment is scaled by a per-patch gate."""

    def __init__(
        self,
        input_length: int,
        output_length: int,
        hidden: int = 32,
        patch_size: int = 8,
        mean_shift_limit: float = 0.08,
        log_contrast_limit: float = 0.45,
        gate_bias_init: float = 2.0,
    ) -> None:
        super().__init__()
        if patch_size <= 1:
            raise ValueError("patch_size must be greater than one.")
        self.output_length = int(output_length)
        self.patch_size = int(patch_size)
        self.mean_shift_limit = float(mean_shift_limit)
        self.log_contrast_limit = float(log_contrast_limit)
        self.moment_predictor = nn.Sequential(
            nn.Conv2d(input_length + output_length, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3 * output_length, 1),
        )
        nn.init.zeros_(self.moment_predictor[-1].weight)
        nn.init.zeros_(self.moment_predictor[-1].bias)
        self.gate_bias = nn.Parameter(
            torch.full((1, self.output_length, 1, 1), float(gate_bias_init))
        )

    def _patch_pool(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            tensor,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            ceil_mode=True,
            count_include_pad=False,
        )

    def _expand(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return patches.repeat_interleave(self.patch_size, -2).repeat_interleave(
            self.patch_size, -1
        )[..., :height, :width]

    def forward(self, history: torch.Tensor, base: torch.Tensor):
        height, width = base.shape[-2:]
        pooled_context = self._patch_pool(torch.cat((history, base), dim=1))
        raw_shift, raw_log_contrast, raw_gate = self.moment_predictor(
            pooled_context
        ).chunk(3, dim=1)
        patch_shift = self.mean_shift_limit * torch.tanh(raw_shift)
        patch_log_contrast = self.log_contrast_limit * torch.tanh(raw_log_contrast)
        patch_contrast = torch.exp(patch_log_contrast)
        patch_gate = torch.sigmoid(raw_gate + self.gate_bias)

        base_patch_mean = self._patch_pool(base)
        base_mean = self._expand(base_patch_mean, height, width)
        shift = self._expand(patch_shift, height, width)
        contrast = self._expand(patch_contrast, height, width)
        gate = self._expand(patch_gate, height, width)
        centered_detail = base - base_mean
        increment = shift + (contrast - 1.0) * centered_detail
        prediction = base + gate * increment
        return prediction, {
            "patch_shift": patch_shift,
            "patch_log_contrast": patch_log_contrast,
            "patch_gate": patch_gate,
            "base_patch_mean": base_patch_mean,
            "base_prediction": base,
            "centered_detail": centered_detail,
        }


class MeteoNetSimVP2OriginalRMSDSEdgeAHGMCRPRGate(nn.Module):
    """CMPAS adapter for the encoder-refined final model."""

    def __init__(
        self,
        input_length: int = 12,
        output_length: int = 6,
        image_size: int | Sequence[int] = 121,
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 2,
        N_T: int = 8,
        mlp_ratio: float = 8.0,
        gsta_kernel_size: int = 21,
        ahg_embed_channels: int = 16,
        ahg_top_k: int = 7,
        ahg_dropout: float = 0.05,
        ahg_residual_scale: float = 0.18,
        ahg_gate_bias: float = -0.85,
        overlap_eps: float = 1e-6,
        encoder_expansion_ratio: int = 1,
        encoder_context_dilation: int = 2,
        encoder_rms_eps: float = 1e-6,
        encoder_beta_init: float = 0.0,
        correction_hidden: int = 32,
        correction_scale: float = 0.25,
        moment_hidden: int = 32,
        patch_size: int = 8,
        mean_shift_limit: float = 0.08,
        log_contrast_limit: float = 0.45,
        gate_bias_init: float = 2.0,
        return_aux: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(image_size, Sequence):
            height, width = int(image_size[0]), int(image_size[1])
        else:
            height = width = int(image_size)
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.return_aux = bool(return_aux)
        self.spatial_divisor = 2 ** sum(_sampling_generator(N_S))
        self.generator = SimVP2OriginalRMSDSEdgeAHGCore(
            in_shape=(input_length, 1, height, width),
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            mlp_ratio=mlp_ratio,
            gsta_kernel_size=gsta_kernel_size,
            ahg_embed_channels=ahg_embed_channels,
            ahg_top_k=ahg_top_k,
            ahg_dropout=ahg_dropout,
            ahg_residual_scale=ahg_residual_scale,
            ahg_gate_bias=ahg_gate_bias,
            overlap_eps=overlap_eps,
            encoder_expansion_ratio=encoder_expansion_ratio,
            encoder_context_dilation=encoder_context_dilation,
            encoder_rms_eps=encoder_rms_eps,
            encoder_beta_init=encoder_beta_init,
        )
        self.residual_correction = SignedResidualCorrection(
            input_length,
            output_length,
            correction_hidden,
            correction_scale,
        )
        self.mcrp = GatedMomentConstrainedResidualProjection(
            input_length,
            output_length,
            hidden=moment_hidden,
            patch_size=patch_size,
            mean_shift_limit=mean_shift_limit,
            log_contrast_limit=log_contrast_limit,
            gate_bias_init=gate_bias_init,
        )

    @property
    def backend(self) -> str:
        return "local"

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        if x.ndim != 4 or x.shape[1] != self.input_length:
            raise ValueError(f"Expected [B,{self.input_length},H,W], got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        pad_h = (self.spatial_divisor - height % self.spatial_divisor) % self.spatial_divisor
        pad_w = (self.spatial_divisor - width % self.spatial_divisor) % self.spatial_divisor
        generator_input = (
            F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            if pad_h or pad_w
            else x
        )
        generated = self.generator(generator_input.unsqueeze(2))
        generated = generated[:, : self.output_length, 0, :height, :width]
        base, residual = self.residual_correction(x, generated)
        prediction, moment_aux = self.mcrp(x, base)
        diagnostics = {
            "generated": generated,
            "residual": residual,
            "base_prediction": base,
            "encoder_refinement": self.generator.encoder_diagnostics(),
            "edge_interaction_ahg": self.generator.ahg.last_diagnostics,
            **moment_aux,
        }
        if return_diagnostics:
            return prediction, diagnostics
        if self.return_aux:
            return {"prediction": prediction, "aux_loss": diagnostics}
        return prediction

HMSimVP = MeteoNetSimVP2OriginalRMSDSEdgeAHGMCRPRGate
MODEL_NAME = "simvp2_original_rms_ds_edge_ahg_mcrp_rgate"
