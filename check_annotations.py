import json

# Load annotations
with open('data/plaque_only/train/_annotations.coco.json', 'r') as f:
    data = json.load(f)

# Print basic stats
print('Number of images:', len(data['images']))
print('Number of annotations:', len(data['annotations']))

# Check category distribution
plaque_count = sum(1 for ann in data['annotations'] if ann['category_id'] == 0)
tooth_count = sum(1 for ann in data['annotations'] if ann['category_id'] == 1)
print('\nCategory distribution:')
print('Plaque annotations:', plaque_count)
print('Tooth annotations:', tooth_count)

# Find images with plaque annotations
images_with_plaque = set()
for ann in data['annotations']:
    if ann['category_id'] == 0:
        images_with_plaque.add(ann['image_id'])

print('\nNumber of images with plaque annotations:', len(images_with_plaque))
