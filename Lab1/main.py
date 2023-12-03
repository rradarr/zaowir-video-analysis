import cv2, cv2.typing
import os
import numpy
import copy
import json
import glob
import random
from pathlib import Path

#pip install opencv-python

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

def main():
    cv2.setNumThreads(12)

    sample_image = Image(".\\s3\\right_20.png")
    image_size = sample_image.bw_data.shape[::-1]

    points_object = []
    points_image = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images = glob.glob(".\\s3\\right_*.png")
    random.shuffle(images)

    image_count = 0
    ret = 1
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


if __name__ == "__main__":
    main()