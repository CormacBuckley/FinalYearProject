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
from timeit import default_timer as timer
import os
import sys
import random
import math
import re
import time
import datetime
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import skimage.draw
from keras.callbacks import TensorBoard
from time import time
import imgaug as ia
from imgaug import augmenters as iaa
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
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1 # Empty + Occupied + Background

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1

    # Add image padding
    IMAGE_PADDING = True

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
config = ParkingConfig()
# config.display()


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
        annotations = [a for a in annotations if a['regions']]

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

            #Could improve this step to benefit efficiency???

        self.add_image(
            "parking",
            image_id=a['filename'],  # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            polygons=polygons,
            names = names)

    def load_mask(self, image_id):
        """Generate instance masks for parking of the given image ID.
        """

        image_info = self.image_info[image_id]
        if image_info["source"] != "parking":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        class_names = info['names']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        class_ids = np.zeros([len(info["polygons"])])

        for i, p in enumerate(class_names):
        #"Type" is the attributes name decided when labeling, etc. 'region_attributes': {Type:'Occupied'}
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

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def im_augs():
        # random example images
    IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/parking/val/40287274793_b5e611b015_o.jpg")
    image = skimage.io.imread(IMAGE_DIR)
    seq =  iaa.OneOf([
    iaa.Multiply((0.2, 1.5)),
    iaa.GaussianBlur(sigma=(1.0, 5.0)),
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    ])
    images_aug = seq.augment_image(image)
    plt.figure()
    plt.imshow(images_aug) 
    plt.show()


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = get_colors_for_class_ids(ids)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, test in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        if label == 'Occupied':
          color = (0, 0, 1)
        else:
          color = (0,1,0)
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

def predictVideo():
    import cv2
    # dataset_train = ParkingDataset()
    # dataset_train.load_parking(args.dataset, "train")
    # dataset_train.prepare()
    video_path="../1008937934-preview.mp4"
    # Video capture
    class_names=['BG', 'Occupied', 'Empty']
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                fps, (width, height))
    start = timer()
    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            # Color splash
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], colors=get_colors_for_class_ids(r['class_ids']))
            frame = display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores']) 
           
            # Add image to video writer
            vwriter.write(frame)
            count += 1
    vwriter.release()
    print("Saved to ", file_name)
    end = timer()
    elapsed = end - start
    
    print("Total time = ", elapsed)
    print("Frames analysed per second = ", count/elapsed)



def predict(weights, loops):

    # dataset_train = ParkingDataset()
    # dataset_train.load_parking(args.dataset, "train")
    # dataset_train.prepare()
    avg = 0
    class_names=['BG', 'Occupied', 'Empty']
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                            config=inference_config,
                            model_dir=MODEL_DIR)
    model_path = (weights)
    model.load_weights(model_path, by_name=True)
    IMAGE_DIR = os.path.join(ROOT_DIR, "datasets/parking/train/IMG_20190218_165352.jpg")

    for i in range (0,loops):
        image = skimage.io.imread(IMAGE_DIR)
        # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(os.listdir(IMAGE_DIR))))

        
        # image = cv2.resize(image,(640,480))
        # Run detection
        # blurer = imgaug.augmenters.GaussianBlur(5.0)
        # image = blurer.augment_image(image)
        start = timer()
        results = model.detect([image], verbose=2) 
        end = timer()
        elapsed = end - start
        avg = avg + elapsed
        r = results[0]
    #     mrcnn = model.run_graph([image], [
    #     ("proposals", model.keras_model.get_layer("ROI").output),
    #     ("probs", model.keras_model.get_layer("mrcnn_class").output),
    #     ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
    #     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    #     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    # ])
    #     det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    #     det_count = np.where(det_class_ids == 0)[0][0]
    #     det_class_ids = det_class_ids[:det_count]

        # print("{} detections: {}".format(
        # det_count, np.array(dataset_train.class_names)[det_class_ids]))
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], colors=get_colors_for_class_ids(r['class_ids']))
        print("Inference Time: ", elapsed)
    print("Average time: ", avg/loops)

        


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
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    """Train the model."""
    # Training dataset.
    dataset_train = ParkingDataset()
    dataset_train.load_parking(args.dataset, "val")
    dataset_train.prepare()

    # Validation dataset
    # dataset_val = ParkingDataset()
    # dataset_val.load_parking(args.dataset, "val")
    # dataset_val.prepare()

    
    # Load and display random samples
    # "2013-04-13_09_15_03#034.jpg"
    image_ids = np.random.choice(dataset_train.image_ids,1)
    for image_id in image_ids:
        print(image_id)
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=5,
    #             layers="heads")
    
    # custom_callbacks=[tensorboard]
    # model.train(dataset_train, dataset_val, 
    #         learning_rate=1e-6 / 10,
    #         epochs=7, 
    #         layers="all",
    #         augmentation = augmentation)

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_parking.h5")
    # model.keras_model.save_weights(model_path)


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
    parser.add_argument('--loops', required=True,
                        metavar="/num/of/loops",
                        help="How many images to predict at once")
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
        predict(args.weights, int(args.loops))
    elif args.command == "aug":
        predictVideo()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'predict'".format(args.command))


