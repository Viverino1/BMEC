import torch
import cv2
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

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

def create_visualization(image, masks, color, thickness=2):
    vis_image = image.copy()
    
    # Create combined mask for pixel counting
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    
    # Draw contours for each individual mask
    for mask in masks:
        mask_binary = mask[0] > 0.5
        combined_mask = combined_mask | mask_binary
        
        # Draw contour for this individual mask
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color.tolist(), thickness)
    
    total_pixels = np.sum(combined_mask)
    return vis_image, combined_mask, total_pixels

def classify_teeth(boxes, image_width):
    """
    Classify all teeth with respect to central incisors.
    """
    # If we have fewer than 4 teeth, we can't reliably identify central incisors
    if len(boxes) < 4:
        return [None] * len(boxes)
    
    # Calculate centroids and areas of each tooth
    centroids = []
    areas = []
    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        
        centroids.append((x_center, y_center))
        areas.append(area)
    
    # Calculate the center point of all teeth
    center_x = sum(c[0] for c in centroids) / len(centroids)
    
    # Separate teeth into top and bottom rows based on y-coordinate
    top_teeth = []
    bottom_teeth = []
    
    # Calculate average y-coordinate to determine the dividing line between top and bottom
    avg_y = sum(c[1] for c in centroids) / len(centroids)
    
    for i, (x, y) in enumerate(centroids):
        if y < avg_y:  # Above the average y-coordinate (top row)
            top_teeth.append((i, x, y, areas[i]))
        else:  # Below the average y-coordinate (bottom row)
            bottom_teeth.append((i, x, y))
    
    # Initialize tooth types
    tooth_types = [None] * len(boxes)
    
    # Find the central incisors in the top row
    top_central_indices = []
    if len(top_teeth) >= 2:
        # Calculate average area of top teeth
        avg_area = sum(t[3] for t in top_teeth) / len(top_teeth)
        
        # Create a score for each tooth based on both position and size
        # Lower score is better (closer to center and larger size)
        tooth_scores = []
        for i, (idx, x, y, area) in enumerate(top_teeth):
            # Distance from center (normalized)
            distance_score = abs(x - center_x) / image_width
            
            # Size score (normalized, inverted so smaller teeth have higher scores)
            # We want larger teeth to have lower scores
            size_score = 0
            if area < avg_area:
                size_score = (avg_area - area) / avg_area
            
            # Combined score (weighted: 60% position, 40% size)
            combined_score = 0.6 * distance_score + 0.4 * size_score
            
            tooth_scores.append((idx, combined_score, x, y))
        
        # Sort by combined score (lower is better)
        tooth_scores.sort(key=lambda t: t[1])
        
        # Label the two teeth with the best scores as central incisors
        for i in range(min(2, len(tooth_scores))):
            idx = tooth_scores[i][0]
            tooth_types[idx] = "CI"
            top_central_indices.append((idx, tooth_scores[i][2], tooth_scores[i][3]))  # Store index, x, y
    
    # Find the bottom central incisors based on their position relative to the top ones
    bottom_central_indices = []
    if len(top_central_indices) == 2 and len(bottom_teeth) >= 2:
        # Calculate the midpoint between the two top central incisors
        top_ci_x = (top_central_indices[0][1] + top_central_indices[1][1]) / 2
        
        # Find the two bottom teeth that are closest to being directly below each top central incisor
        bottom_candidates = []
        
        for idx, x, y in bottom_teeth:
            # Calculate horizontal distance from this tooth to each top central incisor
            dist_to_top_ci_center = abs(x - top_ci_x)
            bottom_candidates.append((idx, dist_to_top_ci_center, x, y))
        
        # Sort by distance (closest first)
        bottom_candidates.sort(key=lambda t: t[1])
        
        # Label the two closest bottom teeth as central incisors
        for i in range(min(2, len(bottom_candidates))):
            idx = bottom_candidates[i][0]
            tooth_types[idx] = "CI"
            bottom_central_indices.append((idx, bottom_candidates[i][2], bottom_candidates[i][3]))  # Store index, x, y
    
    # Now label the rest of the teeth based on their position relative to central incisors
    
    # For top row - label teeth to the left and right of central incisors
    if len(top_central_indices) == 2:
        # Sort top teeth by x-coordinate
        sorted_top_teeth = sorted(top_teeth, key=lambda t: t[1])
        
        # Find indices of top central incisors in the sorted list
        ci_positions = []
        for i, (idx, x, y, _) in enumerate(sorted_top_teeth):
            if idx in [ci[0] for ci in top_central_indices]:
                ci_positions.append(i)
        
        if len(ci_positions) == 2:
            left_ci_pos = min(ci_positions)
            right_ci_pos = max(ci_positions)
            
            # Label teeth to the left of leftmost central incisor
            for i in range(left_ci_pos):
                pos = left_ci_pos - i - 1
                if pos >= 0:
                    idx = sorted_top_teeth[pos][0]
                    if i == 0:
                        tooth_types[idx] = "LI"  # Lateral Incisor
                    elif i == 1:
                        tooth_types[idx] = "CA"  # Canine
                    elif i == 2:
                        tooth_types[idx] = "P1"  # First Premolar
                    elif i == 3:
                        tooth_types[idx] = "P2"  # Second Premolar
                    else:
                        tooth_types[idx] = "MO"  # Molar
            
            # Label teeth to the right of rightmost central incisor
            for i in range(len(sorted_top_teeth) - right_ci_pos - 1):
                pos = right_ci_pos + i + 1
                if pos < len(sorted_top_teeth):
                    idx = sorted_top_teeth[pos][0]
                    if i == 0:
                        tooth_types[idx] = "LI"  # Lateral Incisor
                    elif i == 1:
                        tooth_types[idx] = "CA"  # Canine
                    elif i == 2:
                        tooth_types[idx] = "P1"  # First Premolar
                    elif i == 3:
                        tooth_types[idx] = "P2"  # Second Premolar
                    else:
                        tooth_types[idx] = "MO"  # Molar
    
    # For bottom row - label teeth to the left and right of central incisors
    if len(bottom_central_indices) == 2:
        # Sort bottom teeth by x-coordinate
        sorted_bottom_teeth = sorted(bottom_teeth, key=lambda t: t[1])
        
        # Find indices of bottom central incisors in the sorted list
        ci_positions = []
        for i, (idx, x, y) in enumerate(sorted_bottom_teeth):
            if idx in [ci[0] for ci in bottom_central_indices]:
                ci_positions.append(i)
        
        if len(ci_positions) == 2:
            left_ci_pos = min(ci_positions)
            right_ci_pos = max(ci_positions)
            
            # Label teeth to the left of leftmost central incisor
            for i in range(left_ci_pos):
                pos = left_ci_pos - i - 1
                if pos >= 0:
                    idx = sorted_bottom_teeth[pos][0]
                    if i == 0:
                        tooth_types[idx] = "LI"  # Lateral Incisor
                    elif i == 1:
                        tooth_types[idx] = "CA"  # Canine
                    elif i == 2:
                        tooth_types[idx] = "P1"  # First Premolar
                    elif i == 3:
                        tooth_types[idx] = "P2"  # Second Premolar
                    else:
                        tooth_types[idx] = "MO"  # Molar
            
            # Label teeth to the right of rightmost central incisor
            for i in range(len(sorted_bottom_teeth) - right_ci_pos - 1):
                pos = right_ci_pos + i + 1
                if pos < len(sorted_bottom_teeth):
                    idx = sorted_bottom_teeth[pos][0]
                    if i == 0:
                        tooth_types[idx] = "LI"  # Lateral Incisor
                    elif i == 1:
                        tooth_types[idx] = "CA"  # Canine
                    elif i == 2:
                        tooth_types[idx] = "P1"  # First Premolar
                    elif i == 3:
                        tooth_types[idx] = "P2"  # Second Premolar
                    else:
                        tooth_types[idx] = "MO"  # Molar
    
    return tooth_types

def process_image(image_path):
    # Load models
    teeth_model, plaque_model, device = load_models()
    
    # Read and prepare image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Get teeth predictions
    teeth_masks, teeth_boxes = get_predictions(teeth_model, image, device, confidence_threshold=0.8)
    
    # Classify teeth
    tooth_types = classify_teeth(teeth_boxes, image.shape[1])
    
    # Prepare plaque image with enhancement
    plaque_image = cv2.addWeighted(image, 1.2, image, 0, 10)
    plaque_masks, _ = get_predictions(plaque_model, plaque_image, device, confidence_threshold=0.1, is_plaque=True)
    
    # Create visualizations with pixel counts
    teeth_vis, teeth_mask, teeth_pixels = create_visualization(original_image.copy(), teeth_masks, np.array([0, 255, 0]))
    plaque_vis, plaque_mask, plaque_pixels = create_visualization(original_image.copy(), plaque_masks, np.array([255, 0, 0]))
    
    # Create combined visualization
    combined_vis = original_image.copy()
    
    # Create a separate teeth visualization with labels for teeth_regions.jpg
    teeth_vis_with_labels = teeth_vis.copy()
    
    # Draw individual teeth contours in green on combined_vis (without labels)
    for mask in teeth_masks:
        mask_binary = mask[0] > 0.5
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined_vis, contours, -1, [0, 255, 0], 2)
    
    # In the process_image function, update the section where labels are added to teeth_vis_with_labels:
    
    # Add labels only to teeth_vis_with_labels
    for i, mask in enumerate(teeth_masks):
        mask_binary = mask[0] > 0.5
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add tooth type label
        if i < len(tooth_types) and tooth_types[i] is not None:
            # Get centroid of the tooth
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Get text size to center the label
                text = tooth_types[i]
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_width, text_height = text_size
                
                # Calculate positions for centered text
                text_x = cx - (text_width // 2)
                text_y = cy + (text_height // 2)
                
                # Draw label background
                cv2.rectangle(
                    teeth_vis_with_labels, 
                    (text_x - 5, text_y - text_height - 5), 
                    (text_x + text_width + 5, text_y + 5), 
                    [0, 0, 0], 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    teeth_vis_with_labels, 
                    text, 
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    [255, 255, 255], 
                    1, 
                    cv2.LINE_AA
                )
    
    # Draw individual plaque contours in red
    for mask in plaque_masks:
        mask_binary = mask[0] > 0.5
        mask_uint8 = mask_binary.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined_vis, contours, -1, [255, 0, 0], 2)
    
    # Calculate image dimensions and percentages
    total_image_pixels = image.shape[0] * image.shape[1]
    teeth_percentage = (teeth_pixels / total_image_pixels) * 100
    plaque_percentage = (plaque_pixels / total_image_pixels) * 100
    
    # Print results
    print("\nDetection Results:")
    print(f"Total Image Size: {total_image_pixels} pixels")
    print(f"Teeth Area: {teeth_pixels} pixels ({teeth_percentage:.2f}% of image)")
    print(f"Plaque Area: {plaque_pixels} pixels ({plaque_percentage:.2f}% of image)")
    
    # Calculate overlap
    overlap_mask = teeth_mask & plaque_mask
    overlap_pixels = np.sum(overlap_mask)
    overlap_percentage = (overlap_pixels / teeth_pixels) * 100 if teeth_pixels > 0 else 0
    print(f"Plaque-Teeth Overlap: {overlap_pixels} pixels ({overlap_percentage:.2f}% of teeth area)")
    
    # Print tooth classification results
    print("\nTooth Classification:")
    for i, tooth_type in enumerate(tooth_types):
        print(f"Tooth {i+1}: {tooth_type}")
    
    # Save results - use teeth_vis_with_labels for teeth_regions.jpg
    cv2.imwrite('/Users/vivekmaddineni/Documents/BMEC/results/teeth_regions.jpg', cv2.cvtColor(teeth_vis_with_labels, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/Users/vivekmaddineni/Documents/BMEC/results/plaque_regions.jpg', cv2.cvtColor(plaque_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/Users/vivekmaddineni/Documents/BMEC/results/combined_regions.jpg', cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
    
    return {
        'teeth_pixels': teeth_pixels,
        'plaque_pixels': plaque_pixels,
        'overlap_pixels': overlap_pixels,
        'total_pixels': total_image_pixels,
        'teeth_percentage': teeth_percentage,
        'plaque_percentage': plaque_percentage,
        'overlap_percentage': overlap_percentage,
        'tooth_types': tooth_types
    }

def main():
    test_image_path = 'data/test/IMG_2476_jpeg_jpg.rf.fb80ee4c6bcb464efbd5c166cfdf461f.jpg'
    process_image(test_image_path)
    print("Detection complete! Check the results folder for the output images.")

if __name__ == "__main__":
    main()