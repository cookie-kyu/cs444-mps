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
    Returns
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
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0, 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
    gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
    Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
    and the gt_bbox should again arbitrarilarly be set to all zeros).
    Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
    You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but in order to make your code efficient 
    during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
    This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
    """
    # TODO(student): Complete this function
    # gt_bboxes = anchor
   
    # B, A, _ = anchor.shape
    # gt_clss = torch.zeros((B, A, 1), dtype=torch.int)
    # gt_bboxes = anchor
    
    # for i in range(B):
    #     iou = compute_bbox_iou(anchor[i], bbox[i], dim=1) 
    #     max_iou, max_iou_idxs = torch.max(iou, dim=1)  
        
    #     gt_clss[i, max_iou < 0.4] = 0
    #     gt_bboxes[i, max_iou < 0.4] = 0

    #     ignore_mask = (max_iou >= 0.4) & (max_iou < 0.5)
    #     gt_clss[i, ignore_mask] = -1
    #     gt_bboxes[i, ignore_mask] = 0

    #     fg_mask = (max_iou) >= 0.5
    #     #gt_clss[i, max_iou >= 0.5] = cls[i, max_iou_indices[0]]
    #     gt_clss[i, max_iou >= 0.5] = cls[i, max_iou_idxs[fg_mask]].to(torch.int)
    #     gt_bboxes[i, max_iou >= 0.5] = bbox[i, max_iou_idxs[max_iou>= 0.5]]

    # return gt_clss, gt_bboxes
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
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth object classes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we compute are with respect to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
    We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
    our regression targets more invariant to the size of the original anchor box. So, this means that:
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_w = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle varying sizees of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
       This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
       our regression target would be different. Therefore our formula changes to delta_w = log(gt_bbox_width / anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
       high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
    """
    # TODO(student): Complete this function
    # A, _ = anchors.shape
    # delta_x = np.zeros((A, ))
    # delta_y = np.zeros((A, ))
    # delta_w = np.zeros((A, ))
    # delta_h = np.zeros((A, ))
    # for i in range(A):
    #     gt_bbox_center = [(gt_bboxes[i][0]+ gt_bboxes[i][2])/2 , (gt_bboxes[i][1]+ gt_bboxes[i][3])/2]
    #     anchor_center = [(anchors[i][0]+ anchors[i][2]) / 2,(anchors[i][1]+ anchors[i][3]) / 2]
    #     anchor_width = (anchors[i][2] - anchors[i][0])
    #     anchor_height =  (anchors[i][3] - anchors[i][1])
    #     delta_x[i] = (gt_bbox_center[0] - anchor_center[0]) / anchor_width
    #     delta_y[i] = (gt_bbox_center[1] - anchor_center[1]) / anchor_height

    #     delta_w[i]= np.log(max(gt_bboxes[i][2] - gt_bboxes[i][0],1) /anchor_width)
    #     delta_h[i]= np.log(max(gt_bboxes[i][3] - gt_bboxes[i][1],1) /anchor_height)


    # return torch.stack([torch.tensor(delta_x), torch.tensor(delta_y), torch.tensor(delta_w), torch.tensor(delta_h)], dim=-1)
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
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # TODO(student): Complete this function
    N, _ = boxes.shape
    
    
    width = boxes[:,2] - boxes[:,0]
    height = boxes[:,3] - boxes[:,1]
    x = boxes[:,0] + 0.5 * width
    y = boxes[:,1] + 0.5 * height
    new_center = [x + deltas[:,0]* width, y + deltas[:,1]* height]
    width_shift = torch.exp(deltas[:,2] ) * width / 2.0
    height_shift =  torch.exp(deltas[:,3] ) * height / 2.0
    new_x1 = new_center[0] - width_shift
    new_x2 = new_center[0] + width_shift
    new_y1 = new_center[1] - height_shift
    new_y2 = new_center[1] + height_shift

    new_boxes = torch.stack([new_x1,new_y1, new_x2, new_y2], dim=1)
    return new_boxes
   

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is used to prevent having many boxes that overlap each other. To do this, if multiple boxes overlap each other beyond a
    threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
    first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
    best bbox(the one with the highest score), It will be choosen over all overlapping boxes. Therefore, you should add this bbox to your final 
    resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

    make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
    """
    # TODO(student): Complete this function
    
    # ious = np.zeros((N,N))
  
    #     ##heehehe do things
    # N, _ = bboxes.shape
    # sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    # keep = []
    # while sorted_indices.numel() > 0:
        
    #     current_index = sorted_indices[0].item()
    #     keep.append(current_index)

    #     if sorted_indices.numel() == 1:  
    #         break
    #     current_box = bboxes[current_index].unsqueeze(0) 
    #     others =   bboxes[sorted_indices[1:]]
    #     #ious= []
    #     # for i in range(N - 1):
            
    #     #     ious.append(compute_bbox_iou(current_box,others[i]))
    #     # ious = torch.tensor(ious)
    #     ious = torch.tensor([compute_bbox_iou(current_box, box) for box in others])

    #     remaining_indices = sorted_indices[1:][ious <= threshold]
        
    #     sorted_indices = remaining_indices

    # return torch.tensor(keep)
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

    return torch.tensor(keep, device=bboxes.device)  
