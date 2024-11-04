import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Set device globally if you have one
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns:
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    B, A, _ = anchor.shape
    gt_clss = torch.zeros((B, A, 1), dtype=torch.int, device=anchor.device)  # Move to the same device as anchor
    gt_bboxes = anchor.to(anchor.device)  # Ensure gt_bboxes is on the same device

    for i in range(B):
        iou = compute_bbox_iou(anchor[i], bbox[i], dim=1) 
        max_iou, max_iou_idxs = torch.max(iou, dim=1)  
        
        gt_clss[i, max_iou < 0.4] = 0
        gt_bboxes[i, max_iou < 0.4] = 0

        ignore_mask = (max_iou >= 0.4) & (max_iou < 0.5)
        gt_clss[i, ignore_mask] = -1
        gt_bboxes[i, ignore_mask] = 0

        fg_mask = (max_iou) >= 0.5
        gt_clss[i, max_iou >= 0.5] = cls[i, max_iou_idxs[fg_mask]].to(torch.int).to(anchor.device)  # Ensure type and device consistency
        gt_bboxes[i, max_iou >= 0.5] = bbox[i, max_iou_idxs[max_iou >= 0.5]]

    return gt_clss, gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    A, _ = anchors.shape
    delta_x = torch.zeros((A,), device=anchors.device)
    delta_y = torch.zeros((A,), device=anchors.device)
    delta_w = torch.zeros((A,), device=anchors.device)
    delta_h = torch.zeros((A,), device=anchors.device)

    for i in range(A):
        gt_bbox_center = [(gt_bboxes[i][0] + gt_bboxes[i][2]) / 2, (gt_bboxes[i][1] + gt_bboxes[i][3]) / 2]
        anchor_center = [(anchors[i][0] + anchors[i][2]) / 2, (anchors[i][1] + anchors[i][3]) / 2]
        anchor_width = (anchors[i][2] - anchors[i][0])
        anchor_height = (anchors[i][3] - anchors[i][1])
        delta_x[i] = (gt_bbox_center[0] - anchor_center[0]) / anchor_width
        delta_y[i] = (gt_bbox_center[1] - anchor_center[1]) / anchor_height

        delta_w[i] = torch.log(torch.max(gt_bboxes[i][2] - gt_bboxes[i][0], torch.tensor(1.0, device=anchors.device)) / anchor_width)
        delta_h[i] = torch.log(torch.max(gt_bboxes[i][3] - gt_bboxes[i][1], torch.tensor(1.0, device=anchors.device)) / anchor_height)

    return torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
    """
    N, _ = boxes.shape
    
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    x = boxes[:, 0] + 0.5 * width
    y = boxes[:, 1] + 0.5 * height
    new_center = [x + deltas[:, 0] * width, y + deltas[:, 1] * height]
    width_shift = torch.exp(deltas[:, 2]) * width / 2.0
    height_shift = torch.exp(deltas[:, 3]) * height / 2.0
    new_x1 = new_center[0] - width_shift
    new_x2 = new_center[0] + width_shift
    new_y1 = new_center[1] - height_shift
    new_y2 = new_center[1] + height_shift

    new_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
    return new_boxes

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    """
    N, _ = bboxes.shape
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    keep = []
    while sorted_indices.numel() > 0:
        
        current_index = sorted_indices[0].item()
        keep.append(current_index)

        if sorted_indices.numel() == 1:  
            break
        current_box = bboxes[current_index].unsqueeze(0) 
        others = bboxes[sorted_indices[1:]]
        
        ious = torch.tensor([compute_bbox_iou(current_box, box.unsqueeze(0)) for box in others], device=bboxes.device)
        remaining_indices = sorted_indices[1:][ious <= threshold]
        
        sorted_indices = remaining_indices

    return torch.tensor(keep, device=bboxes.device)  # Ensure the return is on the same device
