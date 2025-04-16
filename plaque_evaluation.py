import os
import cv2
import json
import numpy as np
from utils.evaluation_metrics import calculate_iou

def load_coco_annotations(annotation_path):
    """
    Load COCO format annotations from file
    """
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def get_plaque_ground_truth_masks(coco_data, image_id):
    """
    Get ground truth masks for plaque from COCO annotations
    
    Args:
        coco_data: COCO format annotation data
        image_id: ID of the image to get masks for
        
    Returns:
        plaque_mask: Binary mask for plaque (0 or 1)
    """
    # Find the image
    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    
    # Get all annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Create empty mask
    height, width = image_info['height'], image_info['width']
    plaque_mask = np.zeros((height, width), dtype=np.uint8)
    
    for ann in annotations:
        # Convert segmentation to binary mask
        seg = ann['segmentation'][0]
        if len(seg) < 6:  # Skip invalid segmentations
            continue
            
        points = np.array(seg).reshape(-1, 2)
        points = points.astype(np.int32)
        
        # Draw the mask
        if ann['category_id'] == 1:  # Plaque
            cv2.drawContours(plaque_mask, [points], -1, 1, -1)
    
    return plaque_mask

def evaluate_plaque_model(image_path, target_plaque_mask):
    """
    Evaluate plaque model on a single image
    
    Args:
        image_path: Path to the input image
        target_plaque_mask: Ground truth mask for plaque (0 or 1)
        
    Returns:
        dict containing IoU score and predicted mask
    """
    from utils.evaluation_metrics import load_models, get_predictions
    
    # Load models
    _, plaque_model, device = load_models()
    
    # Read and prepare image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    plaque_masks, _ = get_predictions(plaque_model, image, device, confidence_threshold=0.1)
    
    # Combine all predicted masks
    pred_plaque_mask = np.zeros_like(target_plaque_mask)
    
    for mask in plaque_masks:
        pred_plaque_mask = np.logical_or(pred_plaque_mask, mask[0] > 0.5)
    
    # Calculate IoU score
    plaque_iou = calculate_iou(pred_plaque_mask, target_plaque_mask)
    
    return {
        'plaque_iou': plaque_iou,
        'plaque_mask': pred_plaque_mask
    }

def main():
    # Paths
    train_dir = "data/plaque_only/train"
    annotation_path = os.path.join(train_dir, "_annotations.coco.json")
    
    # Load COCO annotations
    coco_data = load_coco_annotations(annotation_path)
    
    # Check if there are any plaque annotations
    plaque_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] == 0]
    
    if not plaque_annotations:
        print("No plaque annotations found in the dataset.")
        print("The dataset only contains tooth annotations.")
        print(f"Total images: {len(coco_data['images'])}")
        print(f"Total tooth annotations: {len(coco_data['annotations'])}")
        return
    
    # Find images with plaque annotations
    images_with_plaque = []
    for image in coco_data['images']:
        image_id = image['id']
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        if any(ann['category_id'] == 0 for ann in annotations):  # Check for plaque annotations
            images_with_plaque.append(image)
    
    print(f"Found {len(images_with_plaque)} images with plaque annotations")
    
    # Evaluate each image with plaque
    for image_info in images_with_plaque:
        image_id = image_info['id']
        image_path = os.path.join(train_dir, image_info['file_name'])
        
        # Get ground truth mask
        plaque_mask = get_plaque_ground_truth_masks(coco_data, image_id)
        
        # Evaluate the image
        results = evaluate_plaque_model(image_path, plaque_mask)
        
        print(f"\nEvaluation Results for {image_info['file_name']}")
        print(f"Plaque IoU: {results['plaque_iou']:.4f}")

if __name__ == "__main__":
    main()
