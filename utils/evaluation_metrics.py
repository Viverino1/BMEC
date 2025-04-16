import torch
import cv2
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

def calculate_iou(pred_mask, target_mask):
    """
    Calculate Intersection over Union (IoU) for a single mask.
    
    Args:
        pred_mask: Predicted mask tensor (H, W)
        target_mask: Ground truth mask tensor (H, W)
    
    Returns:
        IoU score as a float tensor
    """
    pred_mask = (pred_mask > 0.5).float()
    target_mask = (target_mask > 0.5).float()
    
    intersection = (pred_mask * target_mask).sum()
    union = (pred_mask + target_mask).sum() - intersection
    
    iou = intersection / (union + 1e-6)
    return iou

def calculate_instance_iou(pred_masks, target_masks):
    """
    Calculate IoU for multiple instance masks.
    
    Args:
        pred_masks: List of predicted masks [(H, W)]
        target_masks: List of ground truth masks [(H, W)]
    
    Returns:
        List of IoU scores for each instance
    """
    ious = []
    for pred, target in zip(pred_masks, target_masks):
        iou = calculate_iou(pred, target)
        ious.append(iou)
    return torch.stack(ious)

def calculate_mean_iou(pred_masks, target_masks):
    """
    Calculate mean IoU across all instances.
    
    Args:
        pred_masks: List of predicted masks [(H, W)]
        target_masks: List of ground truth masks [(H, W)]
    
    Returns:
        Mean IoU score as a float tensor
    """
    ious = calculate_instance_iou(pred_masks, target_masks)
    return ious.mean()

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load teeth model
    teeth_model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    teeth_model.roi_heads.box_predictor.cls_score.out_features = 2
    teeth_model.roi_heads.mask_predictor.mask_fcn_logits.out_channels = 2
    teeth_model.load_state_dict(torch.load('/Users/vivekmaddineni/Documents/BMEC/models/teeth_model.pth', map_location=device))
    teeth_model.to(device)
    teeth_model.eval()
    
    # Load plaque model
    plaque_model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    plaque_model.roi_heads.box_predictor.cls_score.out_features = 2
    plaque_model.roi_heads.mask_predictor.mask_fcn_logits.out_channels = 2
    plaque_model.load_state_dict(torch.load('/Users/vivekmaddineni/Documents/BMEC/models/plaque_model.pth', map_location=device))
    plaque_model.to(device)
    plaque_model.eval()
    
    return teeth_model, plaque_model, device

def get_predictions(model, image, device, confidence_threshold, is_plaque=False):
    image_tensor = F.to_tensor(image)
    
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
    
    masks = prediction[0]['masks'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    
    mask_filter = scores >= confidence_threshold
    masks = masks[mask_filter]
    boxes = boxes[mask_filter]
    
    return masks, boxes

def calculate_iou(pred_mask, target_mask):
    """Calculate IoU between predicted and target masks"""
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    target_mask = (target_mask > 0.5).astype(np.uint8)
    
    intersection = np.sum(pred_mask * target_mask)
    union = np.sum(pred_mask + target_mask) - intersection
    
    return intersection / (union + 1e-6)

def evaluate_single_image(image_path, target_teeth_mask):
    """
    Evaluate teeth model on a single image and calculate IoU
    
    Args:
        image_path: Path to the input image
        target_teeth_mask: Ground truth mask for teeth (0 or 1)
    
    Returns:
        dict containing IoU score for teeth model
    """
    # Load models
    teeth_model, _, device = load_models()
    
    # Read and prepare image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    teeth_masks, _ = get_predictions(teeth_model, image, device, confidence_threshold=0.8)
    
    # Combine all predicted masks for teeth
    pred_teeth_mask = np.zeros_like(target_teeth_mask)
    
    for mask in teeth_masks:
        pred_teeth_mask = np.logical_or(pred_teeth_mask, mask[0] > 0.5)
    
    # Calculate IoU score
    teeth_iou = calculate_iou(pred_teeth_mask, target_teeth_mask)
    
    return {
        'teeth_iou': teeth_iou,
        'teeth_mask': pred_teeth_mask
    }
