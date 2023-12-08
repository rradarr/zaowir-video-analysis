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

from encoder import NpEncoder


class Image:
    base_dir_path = Path.cwd() / "s3"

    file_path: Path
    rgb_data: cv2.typing.MatLike
    bw_data: cv2.typing.MatLike

    def __init__(self, filename: str, dont_load: bool = False):
        self.file_path = Path(filename)
        if not dont_load:
            self.load_img()

    def load_img(self):
        self.rgb_data = cv2.imread(str(self.file_path))
        self.bw_data = cv2.cvtColor(self.rgb_data, cv2.COLOR_BGR2GRAY)

    def show(self):
        cv2.imshow(self.file_path.name, self.rgb_data)
        cv2.waitKey()


def testing():
    image = Image(f"{Path.cwd()}/s3/right_20.png")
    print(numpy.shape(image.rgb_data))
    
    found_chessboard, corners = cv2.findChessboardCorners(image.bw_data, (8, 6))
    if found_chessboard:
        print(f"Chessboard found in {image.file_path.name}")
        cv2.drawChessboardCorners(image.rgb_data, (8, 6), corners, True)
        image.show()
    else:
        print(f"Chessboard not found in {image.file_path.name}")

    cv2.destroyAllWindows()

def get_object_points():
    objp = numpy.zeros((8*6,3), numpy.float32)
    objp[:,:2] = numpy.mgrid[0:8,0:6].T.reshape(-1,2) * 28.67
    return objp

def calibrate_stereo_and_save(out_file_name: str,
                              dataset_left_name_pattern: str,
                              dataset_right_name_pattern: str):
    points_object = []
    points_image_left = []
    points_image_right = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_file_names_left = glob.glob(dataset_left_name_pattern)
    image_file_names_right = glob.glob(dataset_right_name_pattern)

    # Get a sample image to determine the shape of images
    sample_image = Image(image_file_names_left[0])
    image_size = sample_image.bw_data.shape[::-1]

    image_count = 0
    ret_l = 100
    mtx_l = dist_l = rvecs_l = tvecs_l = None
    ret_r = 100
    mtx_r = dist_r = rvecs_r = tvecs_r = None
    for name_left, name_right in zip(image_file_names_left, image_file_names_right):
        # Get the images
        image_left = Image(name_left)
        image_right = Image(name_right)

        # Find patterns
        found_chessboard_left, corners_left = cv2.findChessboardCorners(image_left.bw_data, (8, 6))
        if not found_chessboard_left:
            print(f"Cound not find chessboard in {image_left.file_path.name}")
            continue
        found_chessboard_right, corners_right = cv2.findChessboardCorners(image_right.bw_data, (8, 6))
        if not found_chessboard_right:
            print(f"Cound not find chessboard in {image_right.file_path.name}")
            continue

        # Pattern found in both, calibrate
        print(f"Found chessboard in {image_left.file_path.name} and {image_right.file_path.name}")
        corners2_left = cv2.cornerSubPix(image_left.bw_data, corners_left, (11,11), (-1,-1), criteria)
        corners2_right = cv2.cornerSubPix(image_right.bw_data, corners_right, (11,11), (-1,-1), criteria)
        points_image_left.append(corners2_left)
        points_image_right.append(corners2_right)
        points_object.append(get_object_points())
        image_count += 1

        if image_count % 10 == 0:
            print("Calibrating...")
            new_ret_l, new_mtx_l, new_dist_l, new_rvecs_l, new_tvecs_l = cv2.calibrateCamera(
                points_object, points_image_left, image_size, mtx_l, dist_l, rvecs_l, tvecs_l) # type: ignore
            print(new_ret_l)
            new_ret_r, new_mtx_r, new_dist_r, new_rvecs_r, new_tvecs_r = cv2.calibrateCamera(
                points_object, points_image_right, image_size, mtx_r, dist_r, rvecs_r, tvecs_r) # type: ignore
            print(new_ret_r)
            if new_ret_l > ret_l or new_ret_r > ret_r:
                break
            else:
                ret_l = new_ret_l
                mtx_l = new_mtx_l
                dist_l = new_dist_l
                rvecs_l = new_rvecs_l
                tvecs_l = new_tvecs_l
                ret_r = new_ret_r
                mtx_r = new_mtx_r
                dist_r = new_dist_r
                rvecs_r = new_rvecs_r
                tvecs_r = new_tvecs_r

    ret_stereo, camera_mat_left, distortion_left,\
        camera_mat_right, distortion_right, rotation_mat,\
            translation_vec, essential_mat, fundamental_mat = cv2.stereoCalibrate(points_object,
                                                                                  points_image_left,
                                                                                  points_image_right,
                                                                                  mtx_l, dist_l,
                                                                                  mtx_r, dist_r,
                                                                                  image_size)
    baseline_cm = numpy.linalg.norm(translation_vec) * 0.1
    print(f"Baseline in cm: {baseline_cm}")
    
    print(f"Reprojection err: {ret_stereo}")
    
    results_dict = {
        "ret_stereo": ret_stereo,
        "camera_mat_left": camera_mat_left,
        "distortion_left": distortion_left,
        "camera_mat_right": camera_mat_right,
        "distortion_right": distortion_right,
        "rotation_mat": rotation_mat,
        "translation_vec": translation_vec,
        "essential_mat": essential_mat,
        "fundamental_mat": fundamental_mat,
        "baseline_cm": baseline_cm
    }
    with open(out_file_name, "w") as outfile:
        json.dump(results_dict, outfile, cls=NpEncoder)

def load_and_rectify_stereo(in_file_name: str,
                            dataset_left_name_pattern: str,
                            dataset_right_name_pattern: str):
    image_file_names_left = glob.glob(dataset_left_name_pattern)
    image_file_names_right = glob.glob(dataset_right_name_pattern)

    # Get a sample image to determine the shape of images
    sample_image = Image(image_file_names_left[0])
    image_size = sample_image.bw_data.shape[::-1]

    # Load the calibration data
    with open(in_file_name, "r") as calibration_file:
        calibration_json = json.load(calibration_file)
        camera_mat_left = numpy.array(calibration_json["camera_mat_left"])
        distortion_left = numpy.array(calibration_json["distortion_left"])
        camera_mat_right = numpy.array(calibration_json["camera_mat_right"])
        distortion_right = numpy.array(calibration_json["distortion_right"])
        rotation_mat = numpy.array(calibration_json["rotation_mat"])
        translation_vec = numpy.array(calibration_json["translation_vec"])
        essential_mat = numpy.array(calibration_json["essential_mat"])
        fundamental_mat = numpy.array(calibration_json["fundamental_mat"])

    for name_left, name_right in zip(image_file_names_left, image_file_names_right):
        # Get the images
        image_left = Image(name_left)
        image_right = Image(name_right)

        rect_left, rect_right, proj_left, proj_right, disparity_to_depth_map, roi_left, roi_right = cv2.stereoRectify(
            camera_mat_left, distortion_left, camera_mat_right, distortion_right,
            image_size, rotation_mat, translation_vec
        )
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_mat_left, distortion_left, rect_left, proj_left, image_size, cv2.CV_32FC1 # type: ignore
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_mat_right, distortion_right, rect_right, proj_right, image_size, cv2.CV_32FC1 # type: ignore
        )

        rect_left = cv2.remap(image_left.bw_data, map_left_x, map_left_y, cv2.INTER_LANCZOS4)
        rect_right = cv2.remap(image_right.bw_data, map_right_x, map_right_y, cv2.INTER_LANCZOS4)

        cv2.imwrite(f'{image_left.file_path.stem}_rectified.png', rect_left)
        cv2.imwrite(f'{image_right.file_path.stem}_rectified.png', rect_right)

def remap_test(in_file_name: str,
                dataset_left_name_pattern: str,
                dataset_right_name_pattern: str):
    image_file_names_left = glob.glob(dataset_left_name_pattern)
    image_file_names_right = glob.glob(dataset_right_name_pattern)

    remap_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    # Get a sample image to determine the shape of images
    sample_image = Image(image_file_names_left[0])
    image_size = sample_image.bw_data.shape[::-1]

    # Load the calibration data
    with open(in_file_name, "r") as calibration_file:
        calibration_json = json.load(calibration_file)
        camera_mat_left = numpy.array(calibration_json["camera_mat_left"])
        distortion_left = numpy.array(calibration_json["distortion_left"])
        camera_mat_right = numpy.array(calibration_json["camera_mat_right"])
        distortion_right = numpy.array(calibration_json["distortion_right"])
        rotation_mat = numpy.array(calibration_json["rotation_mat"])
        translation_vec = numpy.array(calibration_json["translation_vec"])
        essential_mat = numpy.array(calibration_json["essential_mat"])
        fundamental_mat = numpy.array(calibration_json["fundamental_mat"])

    for name_left, name_right in zip(image_file_names_left, image_file_names_right):
        # Get the images
        image_left = Image(name_left)
        image_right = Image(name_right)

        rect_left, rect_right, proj_left, proj_right, disparity_to_depth_map, roi_left, roi_right = cv2.stereoRectify(
            camera_mat_left, distortion_left, camera_mat_right, distortion_right,
            image_size, rotation_mat, translation_vec
        )
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_mat_left, distortion_left, rect_left, proj_left, image_size, cv2.CV_32FC1 # type: ignore
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_mat_right, distortion_right, rect_right, proj_right, image_size, cv2.CV_32FC1 # type: ignore
        )

        for remap_method in remap_methods:
            start_time = time.time()
            rect_left = cv2.remap(image_left.bw_data, map_left_x, map_left_y, remap_method)
            rect_right = cv2.remap(image_right.bw_data, map_right_x, map_right_y, remap_method)
            print(f'rm{remap_method}: {round(time.time() - start_time, 5)} seconds')

            cv2.imwrite(f'{image_left.file_path.stem}_rectified_rm{remap_method}.png', rect_left)
            cv2.imwrite(f'{image_right.file_path.stem}_rectified_rm{remap_method}.png', rect_right)

def load_rectified_and_draw_epi(in_file_name: str,
                dataset_left_name_pattern: str,
                dataset_right_name_pattern: str):
    image_file_names_left = glob.glob(dataset_left_name_pattern)
    image_file_names_right = glob.glob(dataset_right_name_pattern)

    # Get a sample image to determine the shape of images
    sample_image = Image(image_file_names_left[0])
    image_size = sample_image.bw_data.shape[::-1]

    # Load the calibration data
    with open(in_file_name, "r") as calibration_file:
        calibration_json = json.load(calibration_file)
        camera_mat_left = numpy.array(calibration_json["camera_mat_left"])
        distortion_left = numpy.array(calibration_json["distortion_left"])
        camera_mat_right = numpy.array(calibration_json["camera_mat_right"])
        distortion_right = numpy.array(calibration_json["distortion_right"])
        rotation_mat = numpy.array(calibration_json["rotation_mat"])
        translation_vec = numpy.array(calibration_json["translation_vec"])
        essential_mat = numpy.array(calibration_json["essential_mat"])
        fundamental_mat = numpy.array(calibration_json["fundamental_mat"])

    for name_left, name_right in zip(image_file_names_left, image_file_names_right):
        # Get the images
        image_left = Image(name_left)
        image_right = Image(name_right)

        rect_left, rect_right, proj_left, proj_right, disparity_to_depth_map, roi_left, roi_right = cv2.stereoRectify(
            camera_mat_left, distortion_left, camera_mat_right, distortion_right,
            image_size, rotation_mat, translation_vec
        )
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            camera_mat_left, distortion_left, rect_left, proj_left, image_size, cv2.CV_32FC1 # type: ignore
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            camera_mat_right, distortion_right, rect_right, proj_right, image_size, cv2.CV_32FC1 # type: ignore
        )

        rect_left = cv2.remap(image_left.bw_data, map_left_x, map_left_y, cv2.INTER_LANCZOS4)
        rect_right = cv2.remap(image_right.bw_data, map_right_x, map_right_y, cv2.INTER_LANCZOS4)

        cv2.rectangle(rect_left, (roi_left[0], roi_left[1]), (roi_left[2], roi_left[3]), (255, 0, 0), 4)
        cv2.rectangle(rect_right, (roi_right[0], roi_right[1]), (roi_right[2], roi_right[3]), (255, 0, 0), 4)

        images_together = numpy.concatenate((rect_left, rect_right), axis=1)
        add_lines(images_together)

        cv2.imwrite(f'{image_right.file_path.stem}_epi_joined.png', images_together)

def add_lines(image):
    for line in range(0, image.shape[1], 50):
        cv2.line(image, (0, line), (image.shape[1], line), (0, 0, 255), 2)


if __name__ == "__main__":
    #calibrate_stereo_and_save("stereo_params.json", ".\\s3\\left_*.png", ".\\s3\\right_*.png")

    #load_and_rectify_stereo("stereo_params.json", ".\\s3\\left_255.png", ".\\s3\\right_255.png")

    #remap_test("stereo_params.json", ".\\s3\\left_255.png", ".\\s3\\right_255.png")

    load_rectified_and_draw_epi("stereo_params.json", ".\\s3\\left_15.png", ".\\s3\\right_15.png")