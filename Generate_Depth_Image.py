import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os

from api import PRN
import utils.depth_image as DepthImage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib = True, is_opencv = False) 

# path_image = './TestImages/0.jpg'

image_folder = './TestImages/'
image_names = os.listdir(image_folder)
image_path = [os.path.join(image_folder, name) for name in image_names]

output_folder = './TestImages/depth/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for idx, path in enumerate(image_path):
    image = imread(path)
    image_shape = [image.shape[0], image.shape[1]]

    pos, is_flip = prn.process(image, None, None, image_shape)

    kpt = prn.get_landmarks(pos)

    # 3D vertices
    vertices = prn.get_vertices(pos)

    depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

    if is_flip:
        # flip vertical depth image
        depth_scene_map = np.flip(depth_scene_map, axis=0)

    # write dense scene map
    cv2.imwrite(os.path.join(output_folder, image_names[idx]), depth_scene_map)
