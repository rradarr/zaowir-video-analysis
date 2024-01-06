import time
from typing import List, Tuple
import cv2, cv2.typing
import os
import numpy
import copy
import json
import glob
import random
from pathlib import Path
from matplotlib import pyplot

from labs.shared.encoder import NpEncoder
from labs.shared.image import Image
from labs.shared.calibration_helpers import get_object_points

class DisparitySearch:
    _image_left: Image
    _image_right: Image

    pattern_size: Tuple[int, int] = (10, 10)
    search_window_size: Tuple[int, int] = (270, 1)

    def __init__(self, img_left_name: str, img_right_name: str, img_scale_factor: float = 1):
        # Scale the search window with the image
        self.search_window_size = (int(self.search_window_size[0] * img_scale_factor), self.search_window_size[1])

        self.set_images(img_left_name, img_right_name)
        self._image_left.bw_data = cv2.resize(self._image_left.bw_data, (0, 0), fx=img_scale_factor, fy=img_scale_factor)
        self._image_right.bw_data = cv2.resize(self._image_right.bw_data, (0, 0), fx=img_scale_factor, fy=img_scale_factor)
        self._image_left.bw_data = self._image_left.bw_data.astype(numpy.int64)
        self._image_right.bw_data = self._image_right.bw_data.astype(numpy.int64)

    def set_images(self, img_left_name: str, img_right_name: str):
        self._image_left = Image(img_left_name)
        self._image_right = Image(img_right_name)

    def _calculate_pixel_dissimiliarity(self, x_pattern: int, y_pattern: int, x_search: int, y_search: int) -> int:
        """Cacluate how different are the two reagions of the image using a sum of square differences.
        
        The higher the returned value the more different and uncorrelated the regions are.
        The lower is is the more similiar and correlated they are."""
        differences = numpy.subtract(self._image_left.bw_data[y_pattern:y_pattern+self.pattern_size[1], x_pattern:x_pattern+self.pattern_size[0]],
                                 self._image_right.bw_data[y_search:y_search+self.pattern_size[1], x_search:x_search+self.pattern_size[0]])
        return numpy.square(differences).sum()

    def _find_feature_coords_in_right_img(self, x_pattern_coord: int, y_pattern_coord: int) -> Tuple[int, int]:
        """ Return the position in the right image of the pattern from the left.
        
        The returned position is the position within the search window with the highest correlation.
        The input and output coordinates are of the top left corner of the patterns."""
        
        x_search_area_size = self.search_window_size[0]
        y_search_area_size = self.search_window_size[1]
        if x_pattern_coord + x_search_area_size > self._image_right.bw_data.shape[1] - self.pattern_size[0]:
            # If the search area would go outside of image, shrink it so the pattern never goes outside image
            overflow = x_search_area_size + x_pattern_coord - (self._image_right.bw_data.shape[1] - self.pattern_size[0])
            x_search_area_size -= overflow
        if y_pattern_coord + y_search_area_size > self._image_right.bw_data.shape[0] - self.pattern_size[1]:
            # If the search area would go outside of image, shrink it so the pattern never goes outside image
            overflow = y_search_area_size + y_pattern_coord - (self._image_right.bw_data.shape[0] - self.pattern_size[1])
            y_search_area_size -= overflow
        
        min_dissimiliarity = numpy.inf
        most_similiar_coords: Tuple[int, int] = (x_pattern_coord, y_pattern_coord)
        # Iterate over the right image within the search window and find the most similiar location
        for x in range(x_pattern_coord, x_pattern_coord + x_search_area_size):
            for y in range(y_pattern_coord, y_pattern_coord + y_search_area_size):
                dissimiliarity = self._calculate_pixel_dissimiliarity(x_pattern_coord, y_pattern_coord, x, y)
                if dissimiliarity < min_dissimiliarity:
                    min_dissimiliarity = dissimiliarity
                    most_similiar_coords = (x, y)

        return most_similiar_coords

    def find_disparity_map(self):
        disparity_map = numpy.zeros(shape = self._image_left.bw_data.shape)

        assert self._image_left is not None and self._image_right is not None
        # iterate over image1 and find feature in image2
        # X and Y of the image are for some reason flipped (y, x)
        for x in range(self._image_left.bw_data.shape[1] - self.pattern_size[0]):
            print(f"Processing x: {x}")
            for y in range(self._image_left.bw_data.shape[0] - self.pattern_size[1]):
                found_position = self._find_feature_coords_in_right_img(x, y)
                # Calculate distance in pixels
                distance = numpy.sqrt(numpy.square(x - found_position[0]) + numpy.square(y - found_position[1]))
                # Store distance in disparity map
                disparity_map[y, x] = distance
        # save disparity map
        disparity_map_normalized = numpy.divide(disparity_map, disparity_map.max())
        disparity_map_normalized = numpy.multiply(disparity_map_normalized, 255)
        disparity_map_normalized = disparity_map_normalized.astype(numpy.uint8)
        cv2.imwrite('disparity_img.jpg', disparity_map_normalized)
        cv2.imshow("image", disparity_map_normalized)
        cv2.waitKey()

        return disparity_map

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = numpy.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        numpy.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def test_ply():
    print('loading images...')
    image_scale = 0.3

    disparity_reference = cv2.imread(".\\Motorcycle-perfect\\disp0.pfm", -1)
    disparity_reference[numpy.isinf(disparity_reference)] = 0
    disparity_reference = numpy.nan_to_num(disparity_reference, nan=0)
    disparity_reference = cv2.resize(disparity_reference, (0,0), fx=image_scale, fy=image_scale)
    disparity_reference = numpy.add(disparity_reference,1)

    searcher = DisparitySearch(".\\Motorcycle-perfect\\motor_L.png", ".\\Motorcycle-perfect\\motor_R.png", image_scale)
    disparity_custom = searcher.find_disparity_map()
    disparity_custom = disparity_custom.astype(numpy.float32) / 16

    h, w = disparity_reference.shape[:2]
    f = 3979.911
    Q = numpy.array(
        [
            [1, 0, 0, -w/2],
            [0,-1, 0, h/2],
            [0, 0, 0, -f],
            [0, 0, 1/193.001, 0]
        ], dtype=numpy.float32
    )
    points = cv2.reprojectImageTo3D(disparity_custom, Q)
    colors = cv2.resize(searcher._image_left.rgb_data, (0,0), fx=image_scale, fy=image_scale)
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    mask = disparity_custom > disparity_custom.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

def test_depth():
    image_scale = 0.3

    disparity_reference = cv2.imread(".\\Motorcycle-perfect\\disp0.pfm", -1)
    disparity_reference[numpy.isinf(disparity_reference)] = 0
    disparity_reference = numpy.nan_to_num(disparity_reference, nan=0)
    disparity_reference = cv2.resize(disparity_reference, (0,0), fx=image_scale, fy=image_scale)
    disparity_reference = numpy.add(disparity_reference,1)
    disparity_reference = disparity_reference.astype(numpy.float32) / 16

    searcher = DisparitySearch(".\\Motorcycle-perfect\\motor_L.png", ".\\Motorcycle-perfect\\motor_R.png", image_scale)
    disparity_custom = searcher.find_disparity_map()
    disparity_custom = numpy.add(disparity_custom, 1)
    disparity_custom = disparity_custom.astype(numpy.float32) / 16

    depth_coeff = numpy.ones(shape = disparity_reference.shape) * (3979.911 * 193.001)
    depth_reference = numpy.divide(depth_coeff, disparity_reference)
    depth_custom = numpy.divide(depth_coeff, disparity_custom)

    # show the results
    color_map = pyplot.cm.get_cmap('turbo', 8)
    fig, axs = pyplot.subplots(2, figsize=(10, 10))
    axs[0].imshow(depth_custom, cmap=color_map)
    axs[0].set_title('Custom Disparity Map')
    axs[0].axis('off')
    axs[1].imshow(depth_reference, cmap=color_map)
    axs[1].set_title('Reference Disparity Map')
    axs[1].axis('off')
    pyplot.show()

def test_disparity():
    image_scale = 0.3

    disparity_reference = cv2.imread(".\\Motorcycle-perfect\\disp0.pfm", -1)
    disparity_reference[numpy.isinf(disparity_reference)] = 0
    disparity_reference = numpy.nan_to_num(disparity_reference, nan=0)
    disparity_reference = cv2.resize(disparity_reference, (0,0), fx=image_scale, fy=image_scale)

    left_img = Image(".\\Motorcycle-perfect\\motor_L.png")
    right_img = Image(".\\Motorcycle-perfect\\motor_R.png")
    left_img.bw_data = cv2.resize(left_img.bw_data, (0, 0), fx=image_scale, fy=image_scale)
    right_img.bw_data = cv2.resize(right_img.bw_data, (0, 0), fx=image_scale, fy=image_scale)

    searcher = DisparitySearch(".\\Motorcycle-perfect\\motor_L.png", ".\\Motorcycle-perfect\\motor_R.png", image_scale)
    # --- Time my approach ---
    start = time.time()
    disparity_custom = searcher.find_disparity_map()
    duration_custom = time.time() - start

    # --- Time StereoBM ---
    stereoBM = cv2.StereoBM.create(numDisparities=80, blockSize=15)
    start = time.time()
    disparity_bm = stereoBM.compute(left_img.bw_data, right_img.bw_data)
    duration_sbm = time.time() - start

    # --- Time StereoSGBM ---
    stereoSGBM = cv2.StereoSGBM.create(minDisparity=0, numDisparities=80, blockSize=15)
    start = time.time()
    disparity_sgbm = stereoSGBM.compute(left_img.bw_data, right_img.bw_data)
    duration_sgbm = time.time() - start

    print(f"Custom approach: {duration_custom}s")
    print(f"StereoBM approach: {duration_sbm}s")
    print(f"StereoSGBM approach: {duration_sgbm}s")

    error_bm = numpy.mean(numpy.subtract(disparity_reference, disparity_bm) ** 2)
    error_sgbm = numpy.mean(numpy.subtract(disparity_reference, disparity_sgbm) ** 2)
    error_custom = numpy.mean(numpy.subtract(disparity_reference, disparity_custom) ** 2)
    print(f"Estimation Error BM: {error_bm}")
    print(f"Estimation Error SGBM: {error_sgbm}")
    print(f"Estimation Error Custom: {error_custom}")

    disparity_bm_normalized = numpy.divide(disparity_bm, disparity_bm.max())
    disparity_bm_normalized = numpy.multiply(disparity_bm_normalized, 255)
    disparity_bm_normalized = disparity_bm_normalized.astype(numpy.uint8)
    cv2.imwrite('disparity_bm_img.jpg', disparity_bm_normalized)

    disparity_sgbm_normalized = numpy.divide(disparity_sgbm, disparity_sgbm.max())
    disparity_sgbm_normalized = numpy.multiply(disparity_sgbm_normalized, 255)
    disparity_sgbm_normalized = disparity_sgbm_normalized.astype(numpy.uint8)
    cv2.imwrite('disparity_sgbm_img.jpg', disparity_sgbm_normalized)

    # show the results
    color_map = pyplot.cm.get_cmap('turbo', 8)
    fig, axs = pyplot.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(disparity_bm, cmap=color_map)
    axs[0, 0].set_title('StereoBM Disparity Map')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(disparity_sgbm, cmap=color_map)
    axs[0, 1].set_title('StereoSGBM Disparity Map')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(disparity_custom, cmap=color_map)
    axs[1, 0].set_title('Custom Disparity Map')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(disparity_reference, cmap=color_map)
    axs[1, 1].set_title('Reference Disparity Map')
    axs[1, 1].axis('off')
    pyplot.colorbar(axs[1, 1].imshow(disparity_reference, cmap=color_map), ax=axs.ravel().tolist(), shrink=0.5)
    pyplot.show()
