from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from torch import nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

import torch.nn.functional as F
import torch_scatter
from timm.models.layers import DropPath, trunc_normal_

from .model_base import PointTransformerV3
from .module import PointModule
from .postprocess import postprocess_batch
from .structure import Point
from .utils import offset2batch, offset2bincount


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        upcast_attention=True,
    ):
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(channels, channels * 1, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.upcast_attention = upcast_attention
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos(self, qkv: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
        return qkv + q_pos if q_pos is not None else qkv

    def forward(
        self, qkv: torch.Tensor, q_pos: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        H = self.num_heads
        C = self.channels

        q = self.q(self.with_pos(qkv, q_pos))
        kv = self.kv(qkv)

        q = q.reshape(-1, H, C // H)
        k, v = kv.reshape(-1, 2, H, C // H).permute(1, 0, 2, 3).unbind(dim=0)
        
        if self.upcast_attention:
            q = q.float()
            k = k.float()
            v = v.float()
        
        # block-diagonal mask to prevent cross-batch attention
        N = qkv.shape[0]
        B = len(cu_seqlens) - 1
        attn_mask = torch.full((N, N), -1e4, dtype=q.dtype, device=q.device)
        
        for b in range(B):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b+1].item()
            attn_mask[start:end, start:end] = 0.0
        
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
        q_sdpa = q.transpose(0, 1).unsqueeze(0).contiguous()
        k_sdpa = k.transpose(0, 1).unsqueeze(0).contiguous()
        v_sdpa = v.transpose(0, 1).unsqueeze(0).contiguous()
        
        feat = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        
        feat = feat.squeeze(0).transpose(0, 1).reshape(-1, C)
        if self.upcast_attention:
            feat = feat.to(qkv.dtype)
        
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        upcast_attention=True,
    ):
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.upcast_attention = upcast_attention
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        H = self.num_heads
        C = self.channels

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape(-1, H, C // H)
        k = k.reshape(-1, H, C // H)
        v = v.reshape(-1, H, C // H)
        
        if self.upcast_attention:
            q = q.float()
            k = k.float()
            v = v.float()
        
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(q.dtype)
        
        q_sdpa = q.transpose(0, 1).unsqueeze(0).contiguous()
        k_sdpa = k.transpose(0, 1).unsqueeze(0).contiguous()
        v_sdpa = v.transpose(0, 1).unsqueeze(0).contiguous()
        
        feat = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        
        feat = feat.squeeze(0).transpose(0, 1).reshape(-1, C)
        if self.upcast_attention:
            feat = feat.to(self.q_proj.weight.dtype)
        
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.RMSNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        upcast_attention=False,
        use_attn_mask=False,
        attn_mask_eps=1e-6,
        supervise_attn_mask=True,
        is_last_block=False,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.use_attn_mask = use_attn_mask
        self.attn_mask_eps = attn_mask_eps
        self.supervise_attn_mask = supervise_attn_mask
        self.is_last_block = is_last_block

        self.norm1 = norm_layer(channels)
        self.ls1 = (
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.norm_kv = norm_layer(channels)
        self.self_attn = SelfAttentionLayer(
            channels, num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            upcast_attention=upcast_attention,
        )
        self.norm2 = norm_layer(channels)
        self.ls2 = (
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )

        self.cross_attn = CrossAttentionLayer(
            channels, num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            upcast_attention=upcast_attention,
        )
        self.norm3 = norm_layer(channels)
        self.ls3 = (
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if self.use_attn_mask or self.is_last_block:
            self.mask_mlp = MLP(channels, channels, channels)

    @staticmethod
    def with_pos(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return x + pos if pos is not None else x

    def _compute_attn_mask(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ):
        m_k = self.mask_mlp(q)
        e_i = kv
        
        z = torch.matmul(e_i, m_k.t()).t()
        p_hat = torch.sigmoid(z)
        attn_mask = torch.log(p_hat.detach() + self.attn_mask_eps)
        
        B = len(cu_seqlens_kv) - 1
        for b in range(B):
            start_q = cu_seqlens_q[b].item()
            end_q = cu_seqlens_q[b+1].item()
            start_kv = cu_seqlens_kv[b].item()
            end_kv = cu_seqlens_kv[b+1].item()
            attn_mask[start_q:end_q, :start_kv] = -1e4
            attn_mask[start_q:end_q, end_kv:] = -1e4
        
        return attn_mask, z

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        pos_q: torch.Tensor = None,
        pos_k: torch.Tensor = None,
    ):
        kv_n = self.norm_kv(kv.float()).to(kv.dtype)
        
        attn_mask = None
        mask_logits = None
        if self.use_attn_mask and hasattr(self, 'mask_mlp'):
            if self.supervise_attn_mask:
                attn_mask, mask_logits = self._compute_attn_mask(q, kv_n, cu_seqlens_q, cu_seqlens_kv)
            else:
                _, mask_logits = self._compute_attn_mask(q, kv_n, cu_seqlens_q, cu_seqlens_kv)
        
        if self.pre_norm:
            shortcut = q
            q_n = self.norm1(q.float()).to(q.dtype)
            q = self.drop_path(
                self.ls1(
                    self.cross_attn(
                        q=self.with_pos(q_n, pos_q),
                        k=self.with_pos(kv_n, pos_k),
                        v=kv_n,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        attn_mask=attn_mask,
                    )
                )
            )
            q += shortcut

            q_n = self.norm2(q.float()).to(q.dtype)
            q = q + self.drop_path(
                self.ls2(self.self_attn(q_n, pos_q, cu_seqlens_q, max_seqlen_q))
            )

            shortcut = q
            q_n = self.norm3(q.float()).to(q.dtype)
            q = q + self.drop_path(self.ls3(self.mlp(q_n)))
        else:
            q += self.drop_path(
                self.ls1(
                    self.cross_attn(
                        q=self.with_pos(q, pos_q),
                        k=self.with_pos(kv_n, pos_k),
                        v=kv_n,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        attn_mask=attn_mask,
                    )
                )
            )
            q = self.norm1(q.float()).to(q.dtype)

            q += self.drop_path(
                self.ls2(self.self_attn(q, pos_q, cu_seqlens_q, max_seqlen_q))
            )
            q = self.norm2(q.float()).to(q.dtype)

            q += self.drop_path(self.ls3(self.mlp(q)))
            q = self.norm3(q.float()).to(q.dtype)
        return q, mask_logits


class MaskQueryDecoder(nn.Module):
    """Inference-only mask query decoder for panoptic segmentation."""

    __max_seqlen = 0

    def __init__(
        self,
        full_in_channels,
        hidden_channels,
        num_heads,
        num_classes,
        num_queries=32,
        depth=3,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        upcast_attention=True,
        pos_emb=True,
        enc_mode=True,
        query_type: Literal["learned", "superpoint"] = "superpoint",
        mlp_point_proj=False,
        use_stuff_head=False,
        stuff_classes=None,
        supervise_attn_mask=True,
    ):
        super().__init__()
        self.full_in_channels = full_in_channels
        self.mask_channels = hidden_channels
        self.num_classes = num_classes
        self.enc_mode = enc_mode
        self.num_queries = num_queries
        self.use_stuff_head = use_stuff_head
        self.stuff_classes = set(stuff_classes) if stuff_classes is not None else set()
        self.supervise_attn_mask = supervise_attn_mask

        self.query_type = query_type
        if self.query_type == "learned":
            self.query_feat = nn.Embedding(self.num_queries, hidden_channels)
            self.query_embed = nn.Embedding(self.num_queries, hidden_channels)
        self.pos_emb = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        ) if pos_emb else None

        self.blocks = nn.ModuleList([
            Block(
                channels=hidden_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path,
                layer_scale=layer_scale,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pre_norm=pre_norm,
                upcast_attention=upcast_attention,
                use_attn_mask=True,
                attn_mask_eps=1e-6,
                supervise_attn_mask=self.supervise_attn_mask,
                is_last_block=(i == depth - 1),
            )
            for i in range(depth)
        ])

        self.final_norm = norm_layer(hidden_channels)
        self.cls_pred = MLP(hidden_channels, hidden_channels, num_classes + 1) if mlp_point_proj else nn.Linear(hidden_channels, num_classes + 1)
        self.full_point_proj = MLP(full_in_channels, hidden_channels, hidden_channels) if mlp_point_proj else nn.Linear(full_in_channels, hidden_channels)
        
        if self.use_stuff_head:
            self.stuff_head = nn.Sequential(
                nn.Linear(full_in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _max_seqlen(self, seq_len: int) -> int:
        if seq_len > self.__max_seqlen:
            self.__max_seqlen = seq_len
        return self.__max_seqlen

    def _get_queries(self, point: Point) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        batch_size = point.offset.shape[0]
        device = point.feat.device
        max_queries = self.num_queries

        if self.query_type == "learned":
            base_q = self.query_feat.weight
            base_pos = self.query_embed.weight if hasattr(self, "query_embed") else None
            q = base_q.unsqueeze(0).repeat(batch_size, 1, 1)
            pos_q = None
            if base_pos is not None:
                pos_q = base_pos.unsqueeze(0).repeat(batch_size, 1, 1)
            counts = torch.full((batch_size,), max_queries, dtype=torch.int32, device=device)
            valid_mask = torch.ones(batch_size, max_queries, dtype=torch.bool, device=device)
            return q, pos_q, counts, valid_mask

        # superpoint queries
        assert "instance_dbscan" in point.keys()
        labels = point.instance_dbscan.flatten()
        batch_ids = point.batch if "batch" in point else offset2batch(point.offset)
        feats = point.feat

        padded_queries = []
        padded_pos = []
        counts_list = []
        mask_list = []

        for b in range(batch_size):
            sel = batch_ids == b
            feats_b = feats[sel]
            labels_b = labels[sel]

            q_parts = []

            inst_mask = labels_b >= 0
            if inst_mask.any():
                inst_feats = feats_b[inst_mask]
                inst_labels = labels_b[inst_mask]
                _, inv = torch.unique(inst_labels, return_inverse=True, sorted=True)
                inst_queries = torch_scatter.scatter_mean(inst_feats, inv, dim=0)
                q_parts.append(inst_queries)

            stuff_mask = labels_b == -1
            if stuff_mask.any():
                stuff_query = feats_b[stuff_mask].mean(dim=0, keepdim=True)
                q_parts.append(stuff_query)

            if q_parts:
                q_b = torch.cat(q_parts, dim=0)
            else:
                q_b = feats_b.new_zeros((0, feats_b.shape[1]))

            q_b = q_b[:max_queries]
            num_q = q_b.shape[0]
            counts_list.append(num_q)

            pad = max_queries - num_q
            if pad > 0:
                pad_tensor = feats_b.new_zeros((pad, feats_b.shape[1]))
                q_b = torch.cat([q_b, pad_tensor], dim=0)

            padded_queries.append(q_b)
            mask_row = torch.zeros(max_queries, dtype=torch.bool, device=device)
            if num_q > 0:
                mask_row[:num_q] = True
            mask_list.append(mask_row)
            padded_pos.append(torch.zeros_like(q_b))

        q_tensor = torch.stack(padded_queries, dim=0)
        pos_tensor = torch.stack(padded_pos, dim=0)
        counts_tensor = torch.full((batch_size,), max_queries, dtype=torch.int32, device=device)
        valid_mask = torch.stack(mask_list, dim=0)
        return q_tensor, pos_tensor, counts_tensor, valid_mask

    def _filter_stuff_points(self, point: Point, stuff_logits: torch.Tensor) -> Tuple[Point, torch.Tensor, torch.Tensor]:
        stuff_probs = stuff_logits.sigmoid()
        thing_mask = stuff_probs < 0.5
        
        if not thing_mask.any():
            thing_mask = torch.zeros_like(thing_mask, dtype=torch.bool)
            thing_mask[0] = True
        
        point_things = point.copy()
        point_things.feat = point.feat[thing_mask]
        point_things.coord = point.coord[thing_mask]
        
        batch_ids = offset2batch(point.offset)
        batch_ids_things = batch_ids[thing_mask]
        new_counts = torch.bincount(batch_ids_things, minlength=len(point.offset))
        point_things.offset = new_counts.cumsum(dim=0)
        
        return point_things, thing_mask, stuff_probs
    
    def _expand_masks_to_full(self, mask_logits: torch.Tensor, thing_mask: torch.Tensor) -> torch.Tensor:
        Q = mask_logits.shape[0]
        N = thing_mask.shape[0]
        
        mask_logits_full = mask_logits.new_full((Q, N), -1e4)
        mask_logits_full[:, thing_mask] = mask_logits
        
        return mask_logits_full

    def _forward_decoder(self, point: Point):
        point_proj = point.feat
        pos_k = self.pos_emb(point.coord) if self.pos_emb else None
        
        cu_seqlens_kv = torch.cat([
            torch.zeros(1, dtype=point.offset.dtype, device=point.offset.device),
            point.offset,
        ]).int()
        max_seqlen_kv = self._max_seqlen(point.offset[-1])

        q, pos_q, query_counts, query_valid = self._get_queries(point)
        cu_seqlens_q = torch.cat([query_counts.new_zeros(1), query_counts.cumsum(dim=0)])
        max_seqlen_q = int(query_counts.max().item()) if query_counts.numel() > 0 else 0

        q = q.reshape(-1, self.mask_channels)
        pos_q = pos_q.reshape(-1, self.mask_channels) if pos_q is not None else None
        query_valid = query_valid.reshape(-1, 1)
        query_valid_f = query_valid.to(q.dtype)

        final_mask_logits = None
        for blk in self.blocks:
            q, mask_logits = blk(
                q, point_proj,
                cu_seqlens_q, cu_seqlens_kv,
                max_seqlen_q, max_seqlen_kv,
                pos_q, pos_k,
            )
            if mask_logits is not None:
                mask_logits = mask_logits * query_valid_f
                final_mask_logits = mask_logits
            q = q * query_valid_f

        q_norm = self.final_norm(q)
        query_counts_long = query_counts.to(torch.long)
        query_valid_flat = query_valid.squeeze(-1).bool()

        return q_norm, point_proj, final_mask_logits, query_counts_long, query_valid_flat

    def up_cast(self, point):
        if not self.enc_mode:
            return point
        
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def _compute_predictions(self, q_features, mask_logits, point, query_counts, query_valid):
        class_embed = self.cls_pred(q_features)

        pred_masks = []
        pred_cls = []
        pred_logits = []

        C = self.num_classes
        B = point.offset.shape[0]
        counts = offset2bincount(point.offset).to(torch.long)
        query_counts = query_counts.to(torch.long)

        query_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=query_counts.device),
            query_counts.cumsum(dim=0),
        ])
        point_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=counts.device),
            counts.cumsum(dim=0),
        ])

        for b in range(B):
            P_b = counts[b].item()
            q_start = query_offsets[b].item()
            q_end = query_offsets[b + 1].item()
            p_start = point_offsets[b].item()
            p_end = point_offsets[b + 1].item()

            mask_logits_b = mask_logits[q_start:q_end, p_start:p_end]
            cls_b = class_embed[q_start:q_end]
            valid_b = query_valid[q_start:q_end]

            mask_logits_b = mask_logits_b[valid_b]
            cls_b = cls_b[valid_b]

            pred_masks.append(mask_logits_b)
            pred_cls.append(cls_b)

            if mask_logits_b.shape[0] > 0:
                s = mask_logits_b.transpose(0, 1).unsqueeze(-1)
                c = cls_b[:, :C].unsqueeze(0)
                logits_b = torch.logsumexp(s + c, dim=1)
            else:
                logits_b = mask_logits.new_zeros((P_b, C))
            pred_logits.append(logits_b)

        pred_logits = torch.cat(pred_logits, dim=0) if pred_logits else mask_logits.new_zeros((0, C))

        return {
            "pred_masks": pred_masks,
            "pred_logits": pred_cls,
            "seg_logits": pred_logits,
        }

    def forward(self, point: Point):
        point_full = self.up_cast(point)

        stuff_logits = None
        thing_mask = None
        stuff_probs = None
        
        if self.use_stuff_head:
            stuff_logits = self.stuff_head(point_full.feat).squeeze(-1)
            point_for_decoder, thing_mask, stuff_probs = self._filter_stuff_points(point_full, stuff_logits)
            full_point_proj = self.full_point_proj(point_for_decoder.feat)
            point_for_decoder.feat = full_point_proj
        else:
            full_point_proj = self.full_point_proj(point_full.feat)
            point_for_decoder = point_full.copy()
            point_for_decoder.feat = full_point_proj
        
        out_q, _, final_mask_logits, query_counts, query_valid = self._forward_decoder(point_for_decoder)

        if self.use_stuff_head and thing_mask is not None:
            final_mask_logits_full = self._expand_masks_to_full(final_mask_logits, thing_mask)
            outputs = self._compute_predictions(out_q, final_mask_logits_full, point_full, query_counts, query_valid)
        else:
            outputs = self._compute_predictions(out_q, final_mask_logits, point_full, query_counts, query_valid)

        if self.use_stuff_head and stuff_logits is not None:
            outputs["stuff_logits"] = stuff_logits
            outputs["stuff_probs"] = stuff_probs

        point_full.pred_cls = outputs["pred_logits"]
        point_full.pred_masks = outputs["pred_masks"]
        point_full.pred_logits = outputs["seg_logits"]
        point_full.stuff_probs = outputs["stuff_probs"]
        point_full.outputs = outputs

        
        return point_full


class Detector(PointModule):
    def __init__(
        self,
        num_classes,
        full_in_channels,
        hidden_channels,
        num_heads,
        num_queries=32,
        backbone=None,
        depth=3,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        upcast_attention=False,
        pos_emb=True,
        query_type: Literal["learned", "superpoint"] = "learned",
        use_stuff_head=True,
        stuff_classes=None,
        mlp_point_proj=False,
        supervise_attn_mask=True,
        # postprocessing parameters
        stuff_threshold=0.5,
        mask_threshold=0.5,
        conf_threshold=0.5,
        nms_kernel='gaussian',
        nms_sigma=2.0,
        nms_pre=-1,
        nms_max=-1,
        min_points=2,
        fill_uncovered=False,
    ):
        super(Detector, self).__init__()
        if "type" in backbone:
            backbone.pop("type")
        self.backbone = PointTransformerV3(**backbone)

        self.stuff_classes = set(stuff_classes) if stuff_classes is not None else set()
        self.postprocess_cfg = dict(
            stuff_threshold=stuff_threshold,
            mask_threshold=mask_threshold,
            conf_threshold=conf_threshold,
            nms_kernel=nms_kernel,
            nms_sigma=nms_sigma,
            nms_pre=nms_pre,
            nms_max=nms_max,
            min_points=min_points,
            background_class_label=list(self.stuff_classes)[0] if len(self.stuff_classes) > 0 else -1,
            fill_uncovered=fill_uncovered,
        )

        self.decoder = MaskQueryDecoder(
            full_in_channels=full_in_channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_queries=num_queries,
            num_classes=num_classes,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            layer_scale=layer_scale,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            upcast_attention=upcast_attention,
            pos_emb=pos_emb,
            enc_mode=getattr(backbone, 'enc_mode', True),
            query_type=query_type,
            use_stuff_head=use_stuff_head,
            stuff_classes=stuff_classes,
            mlp_point_proj=mlp_point_proj,
            supervise_attn_mask=supervise_attn_mask,
        )

    def forward(self, input_dict, return_point=False):
        """
        Forward pass with postprocessed outputs by default.
        
        Args:
            input_dict: input data dict
            return_point: if True, include the Point object in outputs
            return_raw: if True, include raw seg_logits/pred_masks/pred_logits
        
        Returns:
            dict with:
                instance_labels: (N,) instance IDs, -1 for stuff/uncovered
                class_labels: (N,) class predictions per point
                confidences: (N,) confidence scores per point
                
            If return_point=True, also includes:
                point: the Point object
                
            If return_raw=True, also includes:
                seg_logits: raw segmentation logits
                pred_masks: list of query mask logits per batch
                pred_logits: list of query class logits per batch
        """
        point = Point(input_dict)
        point = self.backbone(point, upcast=False)
        point = self.decoder(point)

        return_dict = dict()
        point_counts = offset2bincount(point.offset).to(torch.long)

        return_dict = {}
        if return_point:
            return_dict["point"] = point
        
        return_dict["seg_logits"] = point.pred_logits
        return_dict["point_counts"] = point_counts
        # return raw outputs for QueryInsSegEvaluator
        if hasattr(point, "outputs") and point.outputs is not None:
            return_dict["pred_logits"] = point.outputs.get("pred_logits")
            return_dict["pred_masks"] = point.outputs.get("pred_masks")
            return_dict["stuff_probs"] = point.outputs.get("stuff_probs")

        return return_dict

    def postprocess(
        self,
        forward_output: dict,
        stuff_threshold: float = None,
        mask_threshold: float = None,
        conf_threshold: float = None,
        nms_kernel: str = None,
        nms_sigma: float = None,
        nms_pre: int = None,
        nms_max: int = None,
        min_points: int = None,
        background_class_label: int = None,
        fill_uncovered: bool = None,
    ):
        cfg = self.postprocess_cfg.copy()
        overrides = {
            'stuff_threshold': stuff_threshold,
            'mask_threshold': mask_threshold,
            'conf_threshold': conf_threshold,
            'nms_kernel': nms_kernel,
            'nms_sigma': nms_sigma,
            'nms_pre': nms_pre,
            'nms_max': nms_max,
            'min_points': min_points,
            'background_class_label': background_class_label,
            'fill_uncovered': fill_uncovered,
        }
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

        return postprocess_batch(
            pred_masks=forward_output["pred_masks"],
            pred_logits=forward_output["pred_logits"],
            stuff_probs=forward_output["stuff_probs"],
            point_counts=forward_output["point_counts"],
            stuff_classes=self.stuff_classes,
            **cfg,
        )