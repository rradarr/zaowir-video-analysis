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
    objp[:,:2] = numpy.mgrid[0:8,0:6].T.reshape(-1,2) * 0.0285
    return objp

def get_all_with_chessboard(name_pattern: str, chessboard_size: Tuple[int, int]) -> List[str]:
    images = glob.glob(name_pattern)
    names_with_chessboard: List[str] = []

    for file in images:
        image = Image(file)
        found_chessboard, corners = cv2.findChessboardCorners(image.bw_data, chessboard_size)
        if found_chessboard:
            names_with_chessboard.append(image.file_path.name)
            print(f"Found cheesboard in {image.file_path.name}")

    return names_with_chessboard

def clibrate_and_save(out_file_name: str, dataset_name_pattern: str, shuffle_dataset: bool = True):
    points_object = []
    points_image = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images = glob.glob(dataset_name_pattern)
    if shuffle_dataset:
        random.shuffle(images)

    # Get a sample image to determine the shape of images
    sample_image = Image(images[0])
    image_size = sample_image.bw_data.shape[::-1]

    image_count = 0
    ret = 100
    mtx = dist = rvecs = tvecs = None
    for file in images:
        image = Image(file)
        print(f"Analyzing image {image.file_path.name}")
        found_chessboard, corners = cv2.findChessboardCorners(image.bw_data, (8, 6))
        if found_chessboard:
            corners2 = cv2.cornerSubPix(image.bw_data, corners, (11,11), (-1,-1), criteria)
            points_image.append(corners2)
            points_object.append(get_object_points())
            image_count += 1

            if image_count % 10 == 0:
                print("Calibrating...")
                new_ret, new_mtx, new_dist, new_rvecs, new_tvecs = cv2.calibrateCamera(
                    points_object, points_image, image_size, mtx, dist, rvecs, tvecs) # type: ignore
                print(new_ret)
                if new_ret > ret:
                    break
                else:
                    ret = new_ret
                    mtx = new_mtx
                    dist = new_dist
                    rvecs = new_rvecs
                    tvecs = new_tvecs
    
    print(f"Return: {ret}\nMatrix: {mtx}\nDistortion: {dist}\nRvecs: {rvecs}\nTvecs: {tvecs}")
    
    results_dict = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
    with open(out_file_name, "w") as outfile:
        json.dump(results_dict, outfile, cls=NpEncoder)

def get_reprojection_error(calibration_file_name: str, dataset_name_pattern: str):
    mean_error = 0
    points_object = []
    points_image = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    with open(calibration_file_name, 'r') as calibration_file:
        calibration_json = json.load(calibration_file)

    image_files = glob.glob(dataset_name_pattern)
    for file in image_files:
        image = Image(file)
        print(f"Analyzing image {image.file_path.name}")
        found_chessboard, corners = cv2.findChessboardCorners(image.bw_data, (8, 6))
        if found_chessboard:
            corners2 = cv2.cornerSubPix(image.bw_data, corners, (11,11), (-1,-1), criteria)
            points_image.append(corners2)
            points_object.append(get_object_points())

    for i in range(len(calibration_json["rvecs"])):
        imgpoints2, _ = cv2.projectPoints(
            points_object[i],
            numpy.array(calibration_json["rvecs"][i]),
            numpy.array(calibration_json["tvecs"][i]),
            numpy.array(calibration_json["mtx"]),
            numpy.array(calibration_json["dist"]))
        error = cv2.norm(points_image[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"Mean reprojection error: {mean_error/len(points_object)}")

def undistort_images(calibration_file_name: str, image_name_pattern: str):
    # load image
    image_file_names = glob.glob(image_name_pattern)

    # load json calibration params
    with open(calibration_file_name, 'r') as calibration_file:
        calibration_json = json.load(calibration_file)

    for file in image_file_names:
        image = Image(file)
        h, w = image.bw_data.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            numpy.array(calibration_json["mtx"]),
            numpy.array(calibration_json["dist"]),
            (w,h),
            1,
            (w,h))
        # undistort
        dst = cv2.undistort(image.bw_data,
            numpy.array(calibration_json["mtx"]),
            numpy.array(calibration_json["dist"]),
            None,
            newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(f'{image.file_path.stem}_undistorted.png', dst)

def remap_images(calibration_file_name: str, image_name_pattern: str):
    # load image
    image_file_names = glob.glob(image_name_pattern)

    # load json calibration params
    with open(calibration_file_name, 'r') as calibration_file:
        calibration_json = json.load(calibration_file)

    for file in image_file_names:
        image = Image(file)
        h, w = image.bw_data.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            numpy.array(calibration_json["mtx"]),
            numpy.array(calibration_json["dist"]),
            (w,h),
            1,
            (w,h))
        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(
            numpy.array(calibration_json["mtx"]),
            numpy.array(calibration_json["dist"]),
            None,
            newcameramtx,
            (w,h),
            5)
        dst = cv2.remap(image.bw_data, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        cv2.imwrite(f'{image.file_path.stem}_remapped.png', dst)

if __name__ == "__main__":
    cv2.setNumThreads(12)

    #get_all_with_chessboard(".\\s3\\left_*.png", (8, 6))

    #clibrate_and_save("with_subpix.json", ".\\s3\\right_*.png", False)

    #get_reprojection_error("with_subpix.json", ".\\s3\\right_*.png")

    #undistort_images("with_subpix.json", ".\\s3\\right_100.png")

    remap_images("with_subpix.json", ".\\s3\\right_100.png")