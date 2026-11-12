"""Event-conditioned forecasters and the paper_new RAIF equations.

The three named time-series baselines are task adaptations: their forecasting
heads produce reference-aligned increments, and a shared-size categorical
conditioning head is added. They do not claim to be the authors' code.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))


def legal_memory(query: Dict[str, Tensor], memory: Dict[str, Tensor]) -> Tensor:
    """A response must be entirely observed strictly before the query date."""
    allowed = memory["end_day"][None, :] < query["day"][:, None]
    if "split" in memory:
        allowed = allowed & (memory["split"][None, :] == 0)
    if "index" in query and "index" in memory:
        allowed = allowed & (query["index"][:, None] != memory["index"][None, :])
    return allowed


class EventEmbedding(nn.Module):
    def __init__(self, n_platforms: int, n_influencers: int, dim: int):
        super().__init__()
        self.platform = nn.Embedding(n_platforms, dim)
        self.influencer = nn.Embedding(n_influencers, dim)

    def forward(self, query: Dict[str, Tensor], disabled: bool = False) -> Tensor:
        platform, influencer = query["platform"], query["influencer"]
        if disabled:
            platform, influencer = torch.zeros_like(platform), torch.zeros_like(influencer)
        return torch.cat((self.platform(platform), self.influencer(influencer)), -1)


class RAIF(nn.Module):
    VARIANTS = {"full", "no_retrieval", "no_response", "no_events", "no_quality", "no_attention"}

    def __init__(self, length: int, features: int, horizon: int, channels: int,
                 n_platforms: int, n_influencers: int, hidden: int = 64,
                 heads: int = 4, key_dim: int = 32, k: int = 5,
                 variant: str = "full", retrieval: str = "response",
                 t_delta: float = 1.0, t_z: float = 0.1, t_r: float = 0.2,
                 lambda_q: float = 1.0, kernel: int = 3, **unused):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError("Unknown RAIF variant: " + variant)
        if hidden % heads or kernel % 2 != 1 or min(t_delta, t_z, t_r) <= 0 or k < 1:
            raise ValueError("Invalid attention, kernel, temperature, or K configuration")
        if retrieval not in {"response", "state", "random", "reverse"}:
            raise ValueError("Invalid retrieval strategy")
        self.horizon, self.channels, self.hidden, self.heads = horizon, channels, hidden, heads
        self.key_dim, self.k, self.variant = key_dim, k, variant
        self.retrieval = "state" if variant == "no_response" else retrieval
        self.t_delta, self.t_z, self.t_r = t_delta, t_z, t_r
        self.lambda_q = 0.0 if variant == "no_quality" else lambda_q
        self.use_quality = variant != "no_quality"
        self.use_retrieval_loss = variant not in {"no_retrieval", "no_response"}
        self.input_projection = nn.Linear(features, hidden, bias=False)
        self.input_position = nn.Parameter(torch.randn(length, hidden) * 0.02)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.encoder_norm = nn.LayerNorm(hidden)
        self.events = EventEmbedding(n_platforms, n_influencers, max(4, hidden // 8))
        self.condition = mlp(hidden + 2 * max(4, hidden // 8), hidden, hidden)
        self.key = nn.Linear(hidden, key_dim, bias=False)
        self.output_position = nn.Parameter(torch.randn(horizon, hidden) * 0.02)
        self.delta_projection = nn.Linear(channels, hidden, bias=False)
        self.key_projection = nn.Linear(key_dim, hidden, bias=False)
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.attn_key = nn.Linear(hidden, hidden, bias=False)
        self.attn_value = nn.Linear(hidden, hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * horizon - 1))
        self.output_norm = nn.LayerNorm(hidden)
        self.response_head = mlp(hidden, hidden, channels)
        self.direct_head = mlp(hidden, hidden, channels)
        self.gate = mlp(2 * hidden, hidden, channels)

    def encode(self, query: Dict[str, Tensor]):
        u = self.input_projection(query["x"]) + self.input_position
        residual = self.conv2(F.gelu(self.conv1(u.transpose(1, 2)))).transpose(1, 2)
        h = self.encoder_norm(u + residual).mean(1)
        c = self.condition(torch.cat((h, self.events(query, self.variant == "no_events")), -1))
        return c, F.normalize(self.key(c), dim=-1, eps=1e-8)

    @torch.no_grad()
    def refresh_memory(self, memory: Dict[str, Tensor], batch_size: int = 512) -> Tensor:
        if len(memory["x"]) == 0:
            return memory["x"].new_empty((0, self.key_dim))
        return torch.cat([self.encode({key: value[start:start + batch_size]
                                      for key, value in memory.items()})[1]
                          for start in range(0, len(memory["x"]), batch_size)])

    def retrieval_loss(self, z: Tensor, memory_z: Tensor, allowed: Tensor,
                       labels: Tensor, memory: Dict[str, Tensor]) -> Tensor:
        """Full legal candidate cross entropy; query labels occur only here."""
        result = z.new_zeros(len(z))
        eligible = allowed.sum(-1) >= 2
        if not eligible.any() or not self.use_retrieval_loss:
            return result
        flat = labels[eligible].flatten(1)
        history = memory["y"].flatten(1)
        distance = (flat.square().sum(-1, keepdim=True) + history.square().sum(-1)[None, :]
                    - 2 * flat @ history.T).clamp_min(0) / history.shape[-1]
        target_logits = -distance / self.t_delta
        if self.use_quality:
            target_logits = target_logits + memory["q"].clamp_min(1e-8).log()[None, :]
        mask = allowed[eligible]
        targets = target_logits.masked_fill(~mask, -1e9).softmax(-1).detach()
        log_probs = (z[eligible] @ memory_z.T / self.t_z).masked_fill(~mask, -1e9).log_softmax(-1)
        result[eligible] = -(targets * log_probs).sum(-1)
        return result

    def _similarity(self, query, memory, z, memory_z):
        if self.retrieval == "state":
            a = F.normalize(query["x"].flatten(1), dim=-1, eps=1e-8)
            b = F.normalize(memory["x"].flatten(1), dim=-1, eps=1e-8)
            return a @ b.T
        return z @ memory_z.T

    def forward(self, query: Dict[str, Tensor], memory: Optional[Dict[str, Tensor]] = None,
                cached_keys: Optional[Tensor] = None, labels: Optional[Tensor] = None):
        c, z = self.encode(query)
        tokens = c[:, None, :] + self.output_position[None, :, :]
        direct = self.direct_head(tokens)
        result = {"prediction": direct, "retrieval_loss": z.new_zeros(len(z)),
                  "selected": torch.empty((len(z), 0), dtype=torch.long, device=z.device),
                  "selected_valid": torch.empty((len(z), 0), dtype=torch.bool, device=z.device),
                  "similarity": z.new_empty((len(z), 0))}
        if self.variant == "no_retrieval" or memory is None or len(memory["x"]) == 0:
            return result
        allowed = legal_memory(query, memory)
        if cached_keys is None:
            cached_keys = self.refresh_memory(memory)
        # Full current-key differentiation for the retrieval objective. The
        # epoch cache is used solely for the nondifferentiable Top-K selection.
        current_keys = self.encode(memory)[1] if labels is not None and self.use_retrieval_loss else None
        if labels is not None and current_keys is not None:
            result["retrieval_loss"] = self.retrieval_loss(z, current_keys, allowed, labels, memory)
        with torch.no_grad():
            ranking = self._similarity(query, memory, z.detach(), cached_keys)
            if self.retrieval == "reverse":
                ranking = -ranking
            elif self.retrieval == "random":
                # Stable pseudorandom order, independent of labels and call order.
                qi = query.get("index", query["day"]).long()[:, None]
                mi = memory.get("index", memory["day"]).long()[None, :]
                ranking = (((qi + 1) * 1103515245 + (mi + 17) * 2654435761)
                           ^ ((qi + 19) * (mi + 31) * 48271)) % 2147483647
                ranking = ranking.to(z.dtype)
            selected = ranking.masked_fill(~allowed, -torch.inf).topk(min(self.k, len(memory["x"])), -1).indices
            valid = allowed.gather(1, selected)
        bsz, count = selected.shape
        if current_keys is None:
            selected_query = {key: value[selected.flatten()] for key, value in memory.items()}
            selected_z = self.encode(selected_query)[1].reshape(bsz, count, -1)
        else:
            selected_z = current_keys[selected]
        if self.retrieval == "state":
            state_q = F.normalize(query["x"].flatten(1), dim=-1, eps=1e-8)
            state_h = F.normalize(memory["x"][selected].flatten(2), dim=-1, eps=1e-8)
            similarity = torch.einsum("bf,bkf->bk", state_q, state_h)
        else:
            similarity = torch.einsum("br,bkr->bk", z, selected_z)
        log_alpha = similarity / self.t_r
        if self.lambda_q:
            log_alpha = log_alpha + self.lambda_q * memory["q"][selected].clamp_min(1e-8).log()
        log_alpha = log_alpha.masked_fill(~valid, -1e9).log_softmax(-1)
        history_y = memory["y"][selected]
        if self.variant == "no_attention":
            weighted = torch.einsum("bk,bkhc->bhc", log_alpha.exp() * valid, history_y)
            output = self.output_norm(tokens + self.delta_projection(weighted))
        else:
            history = (self.delta_projection(history_y) + self.output_position[None, None, :, :]
                       + self.key_projection(selected_z)[:, :, None, :]).flatten(1, 2)
            head_dim = self.hidden // self.heads
            q = self.query_projection(tokens).view(bsz, self.horizon, self.heads, head_dim).transpose(1, 2)
            k = self.attn_key(history).view(bsz, count * self.horizon, self.heads, head_dim).transpose(1, 2)
            v = self.attn_value(history).view(bsz, count * self.horizon, self.heads, head_dim).transpose(1, 2)
            positions = torch.arange(self.horizon, device=z.device)
            relative = positions[:, None] - positions[None, :] + self.horizon - 1
            bias = self.relative_bias[:, relative].repeat(1, 1, count)
            attention = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
            attention = attention + bias[None, :, :, :] + log_alpha.repeat_interleave(self.horizon, -1)[:, None, None, :]
            attention = attention.masked_fill(~valid.repeat_interleave(self.horizon, -1)[:, None, None, :], -1e9)
            attended = (attention.softmax(-1) @ v).transpose(1, 2).reshape(bsz, self.horizon, self.hidden)
            output = self.output_norm(tokens + self.attn_out(attended))
        response = self.response_head(output)
        gate = torch.sigmoid(self.gate(torch.cat((tokens, output), -1)))
        prediction = gate * response + (1 - gate) * direct
        prediction = torch.where(valid.any(-1)[:, None, None], prediction, direct)
        result.update(prediction=prediction, selected=selected, selected_valid=valid, similarity=similarity)
        return result


class DirectForecaster(nn.Module):
    """Common event embedding and a task-specific H x C prediction head."""
    def __init__(self, name: str, length: int, features: int, horizon: int, channels: int,
                 n_platforms: int, n_influencers: int, hidden: int = 64,
                 heads: int = 4, layers: int = 2, dropout: float = 0.1, **unused):
        super().__init__()
        self.name, self.horizon, self.channels = name, horizon, channels
        emb = max(4, hidden // 8)
        self.events = EventEmbedding(n_platforms, n_influencers, emb)
        self.event_output = mlp(2 * emb, hidden, horizon * channels)
        if name == "DLinear":
            self.kernel = min(5, length if length % 2 else length - 1)
            self.trend = nn.Linear(length, horizon)
            self.seasonal = nn.Linear(length, horizon)
            nn.init.constant_(self.trend.weight, 1 / length)
            nn.init.constant_(self.seasonal.weight, 1 / length)
            self.channel_head = nn.Linear(features, channels)
        else:
            layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout,
                                               activation="gelu", batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            if name == "ECT":
                self.input = nn.Linear(features, hidden)
                self.event_token = nn.Linear(2 * emb, hidden)
                self.position = nn.Parameter(torch.randn(length, hidden) * 0.02)
                self.output = mlp(hidden, hidden, horizon * channels)
            elif name == "iTransformer":
                self.input = nn.Linear(length, hidden)
                self.output = nn.Linear(hidden, horizon)
                self.channel_head = nn.Linear(features, channels)
            elif name == "PatchTST":
                self.patch = min(4, length)
                self.stride = max(1, self.patch // 2)
                patches = (length - self.patch) // self.stride + 1
                self.input = nn.Linear(self.patch, hidden)
                self.position = nn.Parameter(torch.randn(patches, hidden) * 0.02)
                self.output = nn.Linear(patches * hidden, horizon)
                self.channel_head = nn.Linear(features, channels)
            else:
                raise ValueError("Unsupported direct forecaster: " + name)

    def native_keys(self, query):
        """Frozen iTransformer tokens mapped by its learned target-channel head."""
        if self.name != "iTransformer":
            raise ValueError("Native target-channel keys require iTransformer")
        tokens = self.transformer(self.input(query["x"].transpose(1, 2)))
        return torch.einsum("cf,bfp->bcp", self.channel_head.weight, tokens)

    def forward(self, query, memory=None, cached_keys=None, labels=None):
        x, event = query["x"], self.events(query)
        if self.name == "DLinear":
            series = x.transpose(1, 2)
            trend = F.avg_pool1d(F.pad(series, (self.kernel // 2,) * 2, mode="replicate"), self.kernel, stride=1)
            out = self.trend(trend) + self.seasonal(series - trend)
            prediction = self.channel_head(out.transpose(1, 2))
        elif self.name == "ECT":
            tokens = self.input(x) + self.position + self.event_token(event)[:, None, :]
            prediction = self.output(self.transformer(tokens).mean(1)).view(len(x), self.horizon, self.channels)
        elif self.name == "iTransformer":
            tokens = self.input(x.transpose(1, 2))
            prediction = self.channel_head(self.output(self.transformer(tokens)).transpose(1, 2))
        else:
            patches = x.transpose(1, 2).unfold(-1, self.patch, self.stride)
            bsz, features, n_patches, patch = patches.shape
            tokens = self.input(patches.reshape(bsz * features, n_patches, patch)) + self.position
            out = self.output(self.transformer(tokens).flatten(1)).view(bsz, features, self.horizon)
            prediction = self.channel_head(out.transpose(1, 2))
        prediction = prediction + self.event_output(event).view(len(x), self.horizon, self.channels)
        return {"prediction": prediction, "retrieval_loss": prediction.new_zeros(len(x))}


class HistoricalMean(nn.Module):
    def forward(self, query, memory=None, cached_keys=None, labels=None):
        if memory is None or not len(memory["x"]):
            raise ValueError("HistoricalMean requires a nonempty training memory")
        allowed = legal_memory(query, memory)
        weights = allowed.to(memory["y"].dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        prediction = torch.einsum("bn,nhc->bhc", weights, memory["y"])
        return {"prediction": prediction, "retrieval_loss": prediction.new_zeros(len(prediction))}


class RAFT(nn.Module):
    """Multiscale Pearson retrieval and linear fusion, adapted to increments.

    Sources: https://arxiv.org/html/2505.04163v1 (Section 3), and the
    independent reference implementation at https://github.com/archon159/RAFT .
    Increment values keep their own level: subtracting future-market offsets
    and restoring current-market levels is inapplicable to reference differences.
    """
    def __init__(self, length, features, horizon, channels, n_platforms, n_influencers,
                 hidden=64, k=5, raft_temperature=0.1, **unused):
        super().__init__()
        self.length, self.horizon, self.channels, self.k = length, horizon, channels, k
        self.temperature = raft_temperature
        self.periods = tuple(period for period in (1, 2, 4) if period <= min(length, horizon))
        self.temporal = nn.Linear(length, horizon)
        self.channel = nn.Linear(features, channels)
        self.offset = nn.Linear(features, channels, bias=False)
        self.period_projection = nn.ModuleList([nn.Linear(horizon // p, horizon) for p in self.periods])
        self.fusion = nn.Linear(2 * horizon, horizon)
        emb = max(4, hidden // 8)
        self.events = EventEmbedding(n_platforms, n_influencers, emb)
        self.event_output = mlp(2 * emb, hidden, horizon * channels)

    def multiscale(self, x):
        keys = []
        for period in self.periods:
            pooled = F.avg_pool1d(x.transpose(1, 2), period, stride=period).transpose(1, 2)
            centered = (pooled - pooled[:, -1:, :]).flatten(1)
            centered = centered - centered.mean(-1, keepdim=True)
            keys.append(F.normalize(centered, dim=-1, eps=1e-8))
        return keys

    @torch.no_grad()
    def refresh_memory(self, memory, batch_size=512):
        return self.multiscale(memory["x"])

    def forward(self, query, memory=None, cached_keys=None, labels=None):
        x = query["x"]
        centered = x - x[:, -1:, :]
        direct = self.channel(self.temporal(centered.transpose(1, 2)).transpose(1, 2))
        direct = direct + self.offset(x[:, -1:, :])
        category = self.event_output(self.events(query)).view(len(x), self.horizon, self.channels)
        if memory is None or len(memory["x"]) == 0:
            return {"prediction": self.fusion(torch.cat((direct, torch.zeros_like(direct)), 1).transpose(1, 2)).transpose(1, 2) + category,
                    "retrieval_loss": x.new_zeros(len(x))}
        allowed = legal_memory(query, memory)
        history_keys = cached_keys if cached_keys is not None else self.refresh_memory(memory)
        retrieved, all_selected, all_valid, all_similarity = [], [], [], []
        for p, qk, hk, projection in zip(self.periods, self.multiscale(x), history_keys, self.period_projection):
            similarities = qk @ hk.T
            selected = similarities.masked_fill(~allowed, -torch.inf).topk(min(self.k, len(hk)), -1).indices
            valid = allowed.gather(1, selected)
            selected_scores = similarities.gather(1, selected)
            weights = (selected_scores / self.temperature).masked_fill(~valid, -1e9).softmax(-1) * valid
            values = F.avg_pool1d(memory["y"].transpose(1, 2), p, stride=p)
            average = torch.einsum("bk,bkch->bch", weights, values[selected])
            retrieved.append(projection(average).transpose(1, 2) * allowed.any(-1)[:, None, None])
            all_selected.append(selected)
            all_valid.append(valid)
            all_similarity.append(selected_scores)
        aggregate = torch.stack(retrieved).sum(0)
        prediction = self.fusion(torch.cat((direct, aggregate), 1).transpose(1, 2)).transpose(1, 2) + category
        selected = torch.cat(all_selected, -1)
        return {"prediction": prediction, "retrieval_loss": x.new_zeros(len(x)),
                "selected": selected, "selected_valid": torch.cat(all_valid, -1),
                "similarity": torch.cat(all_similarity, -1),
                "selection_group": torch.tensor(self.periods, device=x.device).repeat_interleave(all_selected[0].shape[1])[None, :].expand(len(x), -1)}


class RATL(nn.Module):
    """Public-equation RATL adaptation; not an exact unpublished J5 replica.

    Source: https://arxiv.org/html/2609.03937v1 , equations (1)-(15).
    The public source omits J5's complete scalar-feature list/mask. This
    implementation uses its explicitly described query, residual block, Direct
    reference, block identity, and variable identity components, without
    inventing unspecified scalar features. The omitted detail is recorded in
    every run configuration.
    """
    def __init__(self, length, features, horizon, channels, n_platforms, n_influencers,
                 hidden=64, heads=4, layers=2, dropout=0.1, k=5,
                 ratl_block=8, ratl_temperature=1.0, ratl_teacher_temperature=1.0,
                 ratl_gap=11, **unused):
        super().__init__()
        self.horizon, self.channels, self.k = horizon, channels, k
        self.block_size, self.gap = ratl_block, ratl_gap
        self.blocks = math.ceil(horizon / ratl_block)
        self.temperature, self.teacher_temperature = ratl_temperature, ratl_teacher_temperature
        self.base = DirectForecaster("iTransformer", length, features, horizon, channels,
                                     n_platforms, n_influencers, hidden, heads, layers, dropout)
        self.query_encoder = nn.Linear(length * features, hidden)
        self.residual_encoder = nn.Linear(ratl_block, hidden)
        self.direct_encoder = nn.Linear(ratl_block, hidden)
        self.block_embedding = nn.Embedding(self.blocks, hidden)
        self.variable_embedding = nn.Embedding(channels, hidden)
        self.token_encoder = mlp(5 * hidden, hidden, hidden)
        self.set_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=0.0,
                                       activation="gelu", batch_first=True), 1, enable_nested_tensor=False)
        self.score = nn.Linear(hidden, 1)
        self.residual_scale = nn.Parameter(torch.ones(()))
        self.register_buffer("gamma", torch.ones(()))
        self.freeze_base()

    def freeze_base(self):
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def refresh_memory(self, memory, batch_size=512):
        keys, residuals = [], []
        self.base.eval()
        for start in range(0, len(memory["x"]), batch_size):
            batch = {name: values[start:start + batch_size] for name, values in memory.items()}
            keys.append(self.base.native_keys(batch))
            residuals.append(batch["y"] - self.base(batch)["prediction"])
        return {"keys": torch.cat(keys), "residuals": torch.cat(residuals)}

    def forward(self, query, memory=None, cached_keys=None, labels=None):
        with torch.no_grad():
            base = self.base(query)["prediction"]
            keys = self.base.native_keys(query)
        if memory is None or len(memory["x"]) == 0:
            return {"prediction": base + self.residual_scale * 0,
                    "base_prediction": base, "correction": torch.zeros_like(base),
                    "retrieval_loss": base.new_zeros(len(base))}
        cache = cached_keys if cached_keys is not None else self.refresh_memory(memory)
        history_keys, residuals = cache["keys"], cache["residuals"]
        allowed = legal_memory(query, memory) & (memory["end_day"][None, :] < query["day"][:, None] - self.gap)
        with torch.no_grad():
            distance = ((keys[:, None] - history_keys[None, :]) ** 2).mean(-1).transpose(1, 2)
            scores = -distance
            selected = scores.masked_fill(~allowed[:, None, :], -torch.inf).topk(min(self.k, len(history_keys)), -1).indices
            valid = allowed[:, None, :].expand(-1, self.channels, -1).gather(2, selected)
            selected_scores = scores.gather(2, selected)
        bsz, channels, count = selected.shape
        channel_index = torch.arange(channels, device=base.device)[None, :, None]
        candidate = residuals.permute(0, 2, 1)[selected, channel_index]  # B,C,K,H
        direct_weights = (selected_scores / self.temperature).masked_fill(~valid, -1e9).softmax(-1) * valid
        direct = (direct_weights[:, :, :, None] * candidate).sum(2)
        candidate = torch.cat((torch.zeros_like(candidate[:, :, :1]), candidate), 2)
        valid = torch.cat((torch.ones((bsz, channels, 1), device=base.device, dtype=torch.bool), valid), 2)
        padded = F.pad(candidate, (0, self.blocks * self.block_size - self.horizon))
        blocked = padded.view(bsz, channels, count + 1, self.blocks, self.block_size).permute(0, 1, 3, 2, 4)
        direct_blocks = F.pad(direct, (0, self.blocks * self.block_size - self.horizon)).view(bsz, channels, self.blocks, self.block_size)
        groups = channels * self.blocks
        candidates = blocked.reshape(bsz * groups, count + 1, self.block_size)
        candidate_valid = valid[:, :, None, :].expand(-1, -1, self.blocks, -1).reshape(bsz * groups, count + 1)
        if self.training:
            order = torch.rand((bsz * groups, count + 1), device=base.device).argsort(-1)
            candidates = candidates.gather(1, order[:, :, None].expand(-1, -1, self.block_size))
            candidate_valid = candidate_valid.gather(1, order)
        query_token = self.query_encoder(query["x"].flatten(1))[:, None, :].expand(-1, groups, -1).reshape(bsz * groups, -1)
        direct_token = self.direct_encoder(direct_blocks.reshape(bsz * groups, self.block_size))
        block_ids = torch.arange(self.blocks, device=base.device).repeat(channels).repeat(bsz)
        variable_ids = torch.arange(channels, device=base.device).repeat_interleave(self.blocks).repeat(bsz)
        repeated = [tensor[:, None, :].expand(-1, count + 1, -1) for tensor in
                    [query_token, direct_token, self.block_embedding(block_ids), self.variable_embedding(variable_ids)]]
        tokens = self.token_encoder(torch.cat([repeated[0], self.residual_encoder(candidates), *repeated[1:]], -1))
        attended = self.set_attention(tokens, src_key_padding_mask=~candidate_valid)
        logits = self.score(attended).squeeze(-1).masked_fill(~candidate_valid, -1e9)
        log_weights = logits.log_softmax(-1)
        correction_blocks = (log_weights.exp()[:, :, None] * candidates).sum(1) * self.residual_scale
        correction = correction_blocks.view(bsz, channels, self.blocks * self.block_size)[:, :, :self.horizon].transpose(1, 2)
        teacher_loss = base.new_zeros(bsz)
        if labels is not None:
            true_residual = (labels - base).transpose(1, 2)
            true_blocks = F.pad(true_residual, (0, self.blocks * self.block_size - self.horizon)).view(bsz * groups, self.block_size)
            positions = torch.arange(self.block_size, device=base.device)[None, :] + block_ids[:, None] * self.block_size
            temporal_valid = positions < self.horizon
            errors = (((candidates - true_blocks[:, None, :]) ** 2) * temporal_valid[:, None, :]).sum(-1)
            errors = errors / temporal_valid.sum(-1, keepdim=True)
            teacher = (-errors / self.teacher_temperature).masked_fill(~candidate_valid, -1e9).softmax(-1).detach()
            teacher_loss = -(teacher * log_weights).sum(-1).view(bsz, groups).mean(-1)
        return {"prediction": base + self.gamma * correction, "base_prediction": base,
                "correction": correction, "retrieval_loss": teacher_loss,
                "selected": selected.reshape(bsz, -1),
                "selected_valid": allowed[:, None, :].expand(-1, channels, -1).gather(2, selected).reshape(bsz, -1),
                "similarity": selected_scores.reshape(bsz, -1),
                "selection_group": torch.arange(channels, device=base.device).repeat_interleave(count)[None, :].expand(bsz, -1)}


def build_model(name: str, **config):
    if name == "RAIF":
        return RAIF(**config)
    if name == "HistoricalMean":
        return HistoricalMean()
    if name == "RAFT":
        return RAFT(**config)
    if name == "RATL":
        return RATL(**config)
    return DirectForecaster(name=name, **config)
