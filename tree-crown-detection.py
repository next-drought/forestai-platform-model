import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import random
import math
import re
import time
import skimage.draw
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class TreeCrownConfig(Config):
    """Configuration for training on the tree crown dataset.
    Derives from the base Config class and overrides specific values.
    """
    # Give the configuration a recognizable name
    NAME = "tree_crown"
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Tree Crown
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # Number of validation steps to run at the end of every training epoch
    VALIDATION_STEPS = 50
    
    # Learning rate and momentum (as described in the paper)
    LEARNING_RATE = 0.001
    
    # Backbone architecture for feature extraction
    BACKBONE = "resnet101"
    
    # Input image resizing - keep images with their original aspect ratio
    # and enforce a maximum size limit
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    
    # ROIs below this threshold are discarded
    DETECTION_MIN_CONFIDENCE = 0.7

class TreeCrownDataset(utils.Dataset):
    def load_tree_crowns(self, dataset_dir, subset):
        """Load a subset of the Tree Crown dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("tree_crown", 1, "tree_crown")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations
        # LabelMe format (poly format annotations)
        annotations = self.load_labelme_annotations(dataset_dir)
        
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance
            polygons = a['polygons']
            image_path = os.path.join(dataset_dir, a['filename'])
            
            # Load the image
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            self.add_image(
                "tree_crown",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
    
    def load_labelme_annotations(self, dataset_dir):
        """Load LabelMe annotations for tree crown polygons.
        This is specifically designed for the annotation format
        used in the paper with Labelme tool.
        """
        # Implementation would depend on specific format of annotations
        # For this example, assuming we have a JSON file for each image
        # with polygon coordinates for tree crowns
        
        # This is a placeholder that would need to be adapted to actual data format
        # The paper mentions using Labelme annotation tool
        annotations = []
        
        # Scan through all files in the directory
        for filename in os.listdir(dataset_dir):
            if filename.endswith('.json'):  # Labelme annotations are typically JSON
                json_path = os.path.join(dataset_dir, filename)
                
                # Parse the JSON file
                import json
                with open(json_path) as f:
                    data = json.load(f)
                
                # Extract image filename from JSON
                image_filename = data['imagePath']
                
                # Extract polygons - adapt this to match actual Labelme format
                polygons = []
                for shape in data['shapes']:
                    if shape['label'] == 'tree_crown':
                        # Convert points to array format
                        points = np.array(shape['points'], dtype=np.int32)
                        polygons.append(points)
                
                annotations.append({
                    'filename': image_filename,
                    'polygons': polygons
                })
        
        return annotations
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a tree crown dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tree_crown":
            return super(self.__class__, self).load_mask(image_id)
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                       dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p[:, 1], p[:, 0])
            mask[rr, cc, i] = 1
        
        # Return mask, and array of class IDs of each instance
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tree_crown":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def split_image(image_path, output_dir, tile_size=(935, 910)):
    """Split a large Google Earth image into smaller sub-images
    as described in the paper.
    
    Args:
        image_path: Path to the large image
        output_dir: Directory to save the sub-images
        tile_size: Size of the sub-images (width, height)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Calculate number of tiles
    n_h = math.ceil(h / tile_size[1])
    n_w = math.ceil(w / tile_size[0])
    
    print(f"Splitting image of size {w}x{h} into {n_w}x{n_h} tiles")
    
    # Split the image
    count = 0
    for i in range(n_h):
        for j in range(n_w):
            x = j * tile_size[0]
            y = i * tile_size[1]
            
            # Handle edge cases
            x_end = min(x + tile_size[0], w)
            y_end = min(y + tile_size[1], h)
            
            # Extract tile
            tile = img[y:y_end, x:x_end]
            
            # Save tile
            tile_path = os.path.join(output_dir, f"tile_{count:03d}.jpg")
            cv2.imwrite(tile_path, tile)
            count += 1
    
    print(f"Split image into {count} tiles")
    return count

def train_model(config, dataset_dir):
    """Train the Mask R-CNN model for tree crown detection.
    
    Args:
        config: TreeCrownConfig instance
        dataset_dir: Directory containing the dataset
    """
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    
    # Load COCO weights as starting point
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
    ])
    
    # Load training dataset
    dataset_train = TreeCrownDataset()
    dataset_train.load_tree_crowns(dataset_dir, "train")
    dataset_train.prepare()
    
    # Load validation dataset
    dataset_val = TreeCrownDataset()
    dataset_val.load_tree_crowns(dataset_dir, "val")
    dataset_val.prepare()
    
    # Train the model
    # First, train only the heads (as per the paper's approach)
    print("Training network heads")
    model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=5,
               layers='heads')
    
    # Fine-tune all layers
    print("Fine-tuning all layers")
    model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE / 10,
               epochs=10,
               layers='all')
    
    return model

def detect_tree_crowns(model, image_path, output_path):
    """Detect tree crowns in an image and save the result.
    
    Args:
        model: Trained Mask R-CNN model
        image_path: Path to the input image
        output_path: Path to save the output visualization
    """
    # Read the image
    image = skimage.io.imread(image_path)
    
    # Detect tree crowns
    results = model.detect([image], verbose=1)
    r = results[0]
    
    # Visualize results
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        ['BG', 'Tree Crown'], r['scores'],
        title="Tree Crown Detection",
        figsize=(12, 12)
    )
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()
    
    return r

def analyze_results(results):
    """Analyze the detection results to get statistics about tree crowns.
    
    Args:
        results: List of detection results for multiple images
    
    Returns:
        Dictionary with statistics
    """
    total_trees = 0
    total_area = 0
    area_distribution = {}
    
    for r in results:
        n_trees = r['masks'].shape[-1]
        total_trees += n_trees
        
        # Calculate area for each tree crown
        for i in range(n_trees):
            mask = r['masks'][:, :, i]
            area = np.sum(mask) * (0.27**2)  # Convert pixels to m² (0.27m resolution)
            total_area += area
            
            # Update area distribution
            bin_size = 50  # bin size in m²
            bin_idx = int(area / bin_size)
            if bin_idx not in area_distribution:
                area_distribution[bin_idx] = 0
            area_distribution[bin_idx] += 1
    
    # Prepare distribution data for plotting
    area_bins = []
    tree_counts = []
    for bin_idx in sorted(area_distribution.keys()):
        min_area = bin_idx * bin_size
        max_area = (bin_idx + 1) * bin_size
        area_bins.append(f"[{min_area}, {max_area})")
        tree_counts.append(area_distribution[bin_idx])
    
    stats = {
        'total_trees': total_trees,
        'total_area': total_area,
        'area_distribution': {
            'bins': area_bins,
            'counts': tree_counts
        }
    }
    
    return stats

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Train and apply Mask R-CNN for tree crown detection.')
    
    parser.add_argument('--mode', required=True,
                       choices=['train', 'detect', 'split'],
                       help='What action to perform')
    
    parser.add_argument('--input_path', required=False,
                       help='Path to input image or directory')
    
    parser.add_argument('--output_path', required=False,
                       help='Path to output directory or file')
    
    parser.add_argument('--weights', required=False,
                       default='last',
                       help='Path to weights .h5 file or "coco" or "last"')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'train' and not args.input_path:
        print("Please provide --input_path for training data directory")
        return
    
    if args.mode == 'detect' and (not args.input_path or not args.output_path):
        print("Please provide --input_path and --output_path for detection")
        return
    
    if args.mode == 'split' and (not args.input_path or not args.output_path):
        print("Please provide --input_path and --output_path for splitting")
        return
    
    # Configuration for the tree crown model
    config = TreeCrownConfig()
    
    # Perform the requested action
    if args.mode == 'split':
        # Split a large image into tiles
        split_image(args.input_path, args.output_path)
    
    elif args.mode == 'train':
        # Train the model
        train_model(config, args.input_path)
    
    elif args.mode == 'detect':
        # Create model for inference
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
        
        # Load weights
        if args.weights.lower() == "coco":
            model.load_weights(COCO_MODEL_PATH, by_name=True)
        elif args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
            model.load_weights(weights_path, by_name=True)
        else:
            model.load_weights(args.weights, by_name=True)
        
        # Process single image or directory
        if os.path.isdir(args.input_path):
            # Process all images in the directory
            results = []
            for filename in os.listdir(args.input_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(args.input_path, filename)
                    output_path = os.path.join(args.output_path, 
                                              f"result_{os.path.splitext(filename)[0]}.png")
                    print(f"Processing {image_path}")
                    result = detect_tree_crowns(model, image_path, output_path)
                    results.append(result)
            
            # Analyze and report overall statistics
            stats = analyze_results(results)
            print(f"Total trees detected: {stats['total_trees']}")
            print(f"Total crown area: {stats['total_area']:.2f} m²")
            
            # Plot area distribution
            plt.figure(figsize=(12, 6))
            plt.bar(stats['area_distribution']['bins'], 
                   stats['area_distribution']['counts'])
            plt.xlabel('Crown Area (m²)')
            plt.ylabel('Number of Trees')
            plt.title('Distribution of Tree Crown Areas')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_path, 'area_distribution.png'))
            
        else:
            # Process a single image
            result = detect_tree_crowns(model, args.input_path, args.output_path)
            print(f"Detected {result['masks'].shape[-1]} trees")

if __name__ == "__main__":
    main()
