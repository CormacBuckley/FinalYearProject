"""
Mask R-CNN
Train on the parking dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage:

    # Train a new model starting from pre-trained COCO weights
    python3 parking.py train --dataset=/path/to/parking/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 parking.py train --dataset=/path/to/parking/dataset --weights=last
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
# pylint: disable=import-error
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ParkingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "parking"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 # Empty + Occupied + Background

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.65

    # Add image padding
    IMAGE_PADDING = True

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
config = ParkingConfig()
config.display()


############################################################
#  Dataset
############################################################

class ParkingDataset(utils.Dataset):

    def load_parking(self, dataset_dir, subset):
        """Load a subset of the parking dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have Occupied and Empty.
        self.add_class("parking", 1, "Occupied")
        self.add_class("parking", 2, "Empty")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                names = [r['region_attributes'] for r in a['regions']]

            
    

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

        self.add_image(
            "parking",
            image_id=a['filename'],  # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            polygons=polygons,
            names = names)

        # self.add_image(
        #     "Empty",
        #     image_id=a['filename'],  # use file name as a unique image id
        #     path=image_path,
        #     width=width, height=height,
        #     polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for parking of the given image ID.
        """

        image_info = self.image_info[image_id]
        if image_info["source"] != "parking":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        class_names = info['names']
        # print(class_names)
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # # Handle occlusions
        # occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        # for i in range(count-2, -1, -1):
        #     mask[:, :, i] = mask[:, :, i] * occlusion
        #     occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # # Map class names to class IDs.
        # class_ids = np.array([self.class_names.index(s[0]) for s in parking])
        class_ids = np.zeros([len(info["polygons"])])

        for i, p in enumerate(class_names):
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            if p == {'Type': 'Occupied'}:
                class_ids[i] = 1
            elif p == {'Type': 'Empty'}:
                class_ids[i] = 2
            
            class_ids = class_ids.astype(int)
        # return mask.astype(np.bool), class_ids.astype(np.int32)
        return mask.astype(np.bool), class_ids



    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "parking":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def get_ax(self, rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((1, 0, 0.1))
        else:
         colors.append((0.4,0.7,0))
    return colors

def predict():

    dataset_val = ParkingDataset()
    dataset_val.load_parking(args.dataset, "val")
    dataset_val.prepare()
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_parking.h5")
    model.load_weights(model_path, by_name=True)

    IMAGE_DIR = os.path.join(ROOT_DIR, "../TestImages/2012-09-12_11_50_46.jpg")
    image = skimage.io.imread(IMAGE_DIR)

    # Run detection
    results = model.detect([image], verbose=1)

    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                 dataset_val.class_names, r['scores'], colors=get_colors_for_class_ids(r['class_ids']))


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def train(model):
    
    """Train the model."""
    # Training dataset.
    dataset_train = ParkingDataset()
    dataset_train.load_parking(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ParkingDataset()
    dataset_val.load_parking(args.dataset, "val")
    dataset_val.prepare()


    # Load and display random samples
    # "2013-04-13_09_15_03#034.jpg"
    # image_ids = np.random.choice(dataset_train.image_ids,1)
    # for image_id in image_ids:
    #     print(image_id)
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='heads')

    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=60, 
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_parking.h5")
    model.keras_model.save_weights(model_path)


    class InferenceConfig(ParkingConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    model_path = model.find_last()

    # Load trained weights
    # print("Loading weights from ", model_path)
    # model.load_weights(model_path, by_name=True)




    # image_id = random.choice(dataset_val.image_ids)
    # original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
    #                         image_id, use_mini_mask=False)

    # log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
    #                             dataset_train.class_names, figsize=(8, 8))


    # results = model.detect([original_image], verbose=1)

    # r = results[0]
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
    #                             dataset_val.class_names, r['scores'], colors=get_colors_for_class_ids(r['class_ids']), ax= get_ax())



# def color_splash(image, mask):
#     """Apply color splash effect.
#     image: RGB image [height, width, 3]
#     mask: instance segmentation mask [height, width, instance count]

#     Returns result image.
#     """
#     # Make a grayscale copy of the image. The grayscale copy still
#     # has 3 RGB channels, though.
#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#     # Copy color pixels from the original color image where mask is set
#     if mask.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         mask = (np.sum(mask, -1, keepdims=True) >= 1)
#         splash = np.where(mask, image, gray).astype(np.uint8)
#     else:
#         splash = gray.astype(np.uint8)
#     return splash


# def detect_and_color_splash(model, image_path=None, video_path=None):
    # assert image_path or video_path

    # # Image or video?
    # if image_path:
    #     # Run model detection and generate the color splash effect
    #     print("Running on {}".format(args.image))
    #     # Read image
    #     image = skimage.io.imread(args.image)
    #     # Detect objects
    #     r = model.detect([image], verbose=1)[0]
    #     # Color splash
    #     splash = color_splash(image, r['masks'])
    #     # Save output
    #     file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    #     skimage.io.imsave(file_name, splash)
    # elif video_path:
    #     import cv2
    #     # Video capture
    #     vcapture = cv2.VideoCapture(video_path)
    #     width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = vcapture.get(cv2.CAP_PROP_FPS)

    #     # Define codec and create video writer
    #     file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    #     vwriter = cv2.VideoWriter(file_name,
    #                               cv2.VideoWriter_fourcc(*'MJPG'),
    #                               fps, (width, height))

    #     count = 0
    #     success = True
    #     while success:
    #         print("frame: ", count)
    #         # Read next image
    #         success, image = vcapture.read()
    #         if success:
    #             # OpenCV returns images as BGR, convert to RGB
    #             image = image[..., ::-1]
    #             # Detect objects
    #             r = model.detect([image], verbose=0)[0]
    #             # Color splash
    #             splash = color_splash(image, r['masks'])
    #             # RGB -> BGR to save image to video
    #             splash = splash[..., ::-1]
    #             # Add image to video writer
    #             vwriter.write(splash)
    #             count += 1
    #     vwriter.release()
    # print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect parking spots.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/parking/dataset/",
                        help='Directory of the Parking dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ParkingConfig()
    else:
        class InferenceConfig(ParkingConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "predict":
        predict()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))



