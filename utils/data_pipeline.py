import tensorflow as tf
import albumentations as A
import cv2
import numpy as np


TARGET_ASPECT_RATIO = 1.75
TARGET_LONGEST_SIDE = 300
INPUT_DIMS = (int(TARGET_LONGEST_SIDE/TARGET_ASPECT_RATIO),TARGET_LONGEST_SIDE)

with tf.device('/cpu:0'):
  @tf.function
  def augment(input_image):
    transform = A.Compose([
        #  A.RGBShift(r_shift_limit=(0,20), g_shift_limit=(0,20), b_shift_limit=(0,20), p=0.5),
        A.OneOf([
          A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, p=0.3),
          # A.CLAHE(p=0.5),
          A.Emboss(alpha=(0.2, 0.4), strength=(0.2, 0.6),p=0.3),
          A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1), p =0.3),
        ])
    ])

    return transform(image=input_image)['image']

def resize(input_image):
  transform = A.Compose([
      A.LongestMaxSize(max_size=INPUT_DIMS[1], p=1, interpolation=cv2.INTER_LANCZOS4),
      A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5)]),
      A.Resize(INPUT_DIMS[0],INPUT_DIMS[1])
      ])

  return transform(image=input_image)['image']

def encode_y(y):
  if(y=='NILM'):
    return 0
  elif(y=='ASC-US'):
    return 1
  elif(y=='ASC-H'):
    return 2
  elif(y=='LSIL'):
    return 3
  elif(y=='HSIL'):
    return 4
  else:
    return 5

with tf.device('/cpu:0'):
  @tf.function
  def normalize(datapoint):
    """normalizes the input image pixel values to be [0,1] """
    normalized_image = datapoint[0] / 255.0
    
    return normalized_image, datapoint[1]

with tf.device('/cpu:0'):
  @tf.function
  def parse_function(datapoint):
    """
    return a resized and normalized pair of image and mask
    args
      datapoint: a single image and its corresponding segmentation mask

    1. load the image from its path, decode it to jpeg, normalize it to [0,1]
    2. decode the run-length encoding to pixels, then project the mask onto canvas with same size as image
    3. resize both the image and segmentation mask, to math the input size of the network i.e (128,128)
    """
    input_image = tf.io.read_file(datapoint[0])
    input_image = tf.image.decode_image(input_image, expand_animations = False, channels=3)
    input_image = tf.cast(input_image, tf.float32)
    input_image = resize(input_image)

    label = encode_y(datapoint[1])

    return input_image, label