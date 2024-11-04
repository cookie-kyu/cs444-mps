import torch
import math
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
    )


class Anchors(nn.Module):
    def __init__(
        self,
        stride,
        sizes=[4, 4 * math.pow(2, 1 / 3), 4 * math.pow(2, 2 / 3)],
        aspect_ratios=[0.5, 1, 2],
    ):
        """
        Args:
            stride: stride of the feature map relative to the original image
            sizes: list of sizes (sqrt of area) of anchors in units of stride
            aspect_ratios: list of aspect ratios (h/w) of anchors
        __init__ function does the necessary precomputations.
        """
        super(Anchors, self).__init__()
        self.stride = stride
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
        
    def forward(self, x):
        """
        Args:
            x: feature map of shape (B, C, H, W)
        Returns:
            anchors: list of anchors in the format (xmin, ymin, xmax, ymax), giving the shape
            of (B, A*4, H, W) where A is the number of types of anchor boxes we have.
        """
        B, _, H, W = x.shape
    
        anchor_offsets = []
        for size in self.sizes:
            for aspect_ratio in self.aspect_ratios:
                w = size * self.stride * (aspect_ratio**0.5)
                h = size * self.stride / (aspect_ratio**0.5)
                anchor_offsets.append([-w/2, -h/2, w/2, h/2])

        # Create anchor_offsets tensor on the same device as x
        anchor_offsets = torch.tensor(anchor_offsets, device=x.device)  # specify device here

        shifts_x = (torch.arange(0, W, device=x.device) * self.stride)  # specify device
        shifts_y = (torch.arange(0, H, device=x.device) * self.stride)  # specify device
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        shifts = shifts.unsqueeze(1)

        anchors = (shifts + anchor_offsets.view(1, -1, 4)).reshape(H, W, -1)
        anchors = anchors.permute(2, 0, 1).reshape(1, -1, H, W)
        anchors = anchors.repeat(B, 1, 1, 1)
        
        # Ensure anchors are on the same device as x
        return anchors.to(x.device)  # specify device here


class RetinaNet(nn.Module):
    def __init__(self, p67=False, fpn=False,num_anchors=9):
        super(RetinaNet, self).__init__()
        self.resnet = [
            create_feature_extractor(
                resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
                return_nodes={
                    "layer2.3.relu_2": "conv3",
                    "layer3.5.relu_2": "conv4",
                    "layer4.2.relu_2": "conv5",
                },
            )
        ]
        self.resnet[0].eval()
        self.cls_head, self.bbox_head = self.get_heads(10, num_anchors)

        self.p67 = p67
        self.fpn = fpn

        anchors = nn.ModuleList()

        self.p5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            group_norm(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
        )
        self._init(self.p5)
        anchors.append(Anchors(stride=32))

        if self.p67:
            self.p6 = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p6)
            self.p7 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p7)
            anchors.append(Anchors(stride=64))
            anchors.append(Anchors(stride=128))

        if self.fpn:
            self.p4_lateral = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                group_norm(256),
            )
            self.p4 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p4)
            self._init(self.p4_lateral)
            anchors.append(Anchors(stride=16))

            self.p3_lateral = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), group_norm(256)
            )
            self.p3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p3)
            self._init(self.p3_lateral)
            anchors.append(Anchors(stride=8))

        self.anchors = anchors

    def _init(self, modules):
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def to(self, device):
        super(RetinaNet, self).to(device)
        self.anchors.to(device)
        self.resnet[0].to(device)
        return self

    def get_heads(self, num_classes, num_anchors, prior_prob=0.01):
        cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(
                256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            ),
        )
        bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, num_
