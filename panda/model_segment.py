import torch
import torch.nn as nn
from .structure import Point
from .model_base import PointTransformerV3


class Segmenter(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        freeze_backbone=False,
        mlp_head=False,
    ):
        super().__init__()

        if mlp_head:
            self.seg_head = nn.Sequential(
                nn.Linear(backbone_out_channels, backbone_out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(backbone_out_channels // 2, num_classes),
            )
        else:
            self.seg_head = (
                nn.Linear(backbone_out_channels, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        if "type" in backbone:
            backbone.pop("type")
        self.backbone = PointTransformerV3(**backbone)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point, upcast=False)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        feat = point.feat

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point
        return_dict["seg_logits"] = seg_logits
        return return_dict
