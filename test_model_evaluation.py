import os
import cv2
import json
import numpy as np
from utils.evaluation_metrics import evaluate_single_image

def load_coco_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def get_ground_truth_masks(coco_data, image_id):
    # Find the image
    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    
    # Get all annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Create empty mask
    height, width = image_info['height'], image_info['width']
    teeth_mask = np.zeros((height, width), dtype=np.uint8)
    
    for ann in annotations:
        # Convert segmentation to binary mask
        seg = ann['segmentation'][0]
        if len(seg) < 6:  # Skip invalid segmentations
            continue
            
        points = np.array(seg).reshape(-1, 2)
        points = points.astype(np.int32)
        
        # Draw the mask
        if ann['category_id'] == 1:  # Tooth
            cv2.drawContours(teeth_mask, [points], -1, 1, -1)
    
    return teeth_mask

def main():
    # Paths
    train_dir = "data/train"
    annotation_path = os.path.join(train_dir, "annotations.coco.json")
    
    # Load COCO annotations
    coco_data = load_coco_annotations(annotation_path)
    
    # Test with the first image
    image_info = coco_data['images'][0]  # Start with the first image
    image_id = image_info['id']
    image_path = os.path.join(train_dir, image_info['file_name'])
    
    # Get ground truth mask
    teeth_mask = get_ground_truth_masks(coco_data, image_id)
    
    # Evaluate the image
    results = evaluate_single_image(
        image_path=image_path,
        target_teeth_mask=teeth_mask
    )
    
    print(f"\nEvaluation Results for {image_info['file_name']}")
    print(f"Teeth IoU: {results['teeth_iou']:.4f}")

if __name__ == "__main__":
    main()
