import os
import cv2
import random
import matplotlib.pyplot as plt

#To view Labled pictures of the dataset

# Set the data directory paths for images and labels
image_dir = r"C:\Users\HP\OneDrive\Desktop\data\sign_data\images"
label_dir = r"C:\Users\HP\OneDrive\Desktop\data\sign_data\labels"

# List of class names
keys = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 
        'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 
        'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']

# Get all image files
image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
random.shuffle(image_files)

def read_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
        labels = [line.strip().split() for line in lines]
    return labels

def draw_boxes(image, labels):
    for label in labels:
        class_id = int(label[0])  # Class ID from the label file
        class_name = keys[class_id]  # Get class name using the class ID
        x, y, w, h = map(float, label[1:])
        image_height, image_width, _ = image.shape
        x1 = int((x - w / 2) * image_width)
        y1 = int((y - h / 2) * image_height)
        x2 = int((x + w / 2) * image_width)
        y2 = int((y + h / 2) * image_height)
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # Put the class name on the image
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Plot images with bounding boxes and class names
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for ax, image_file in zip(axes.ravel(), image_files[:9]):
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')

    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read labels
    labels = read_labels(label_path)

    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image_rgb, labels)

    # Display image with bounding boxes and set the labels as titles
    ax.imshow(image_with_boxes)
    ax.set_title('Class: ' + str(labels[0][0]) if labels else "No Label")
    ax.axis('off')

plt.tight_layout()
plt.show()

#To view distribution of the data between class

# Set the data directory paths for images and labels
image_dir = r"C:\Users\HP\OneDrive\Desktop\data\sign_data\images"
label_dir = r"C:\Users\HP\OneDrive\Desktop\data\sign_data\labels"

# List of class names
keys = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 
        'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 
        'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']

# Dictionary to count the number of images for each class
class_count = {key: 0 for key in keys}

# Iterate over the label files and count class occurrences
label_files = [file for file in os.listdir(label_dir) if file.endswith('.txt')]
for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)
    labels = read_labels(label_path)
    
    # Count the class occurrences
    for label in labels:
        class_id = int(label[0])  # Class ID from the label file
        if 0 <= class_id < len(keys):
            class_count[keys[class_id]] += 1

# Print out the counts for each class
for class_name, count in class_count.items():
    print(f"{class_name}: {count} images")

