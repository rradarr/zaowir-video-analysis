from typing import List, Tuple
import cv2, cv2.typing
import cv2.aruco
import os
import numpy
import copy
import json
import glob
import random
from pathlib import Path

from labs.shared.encoder import NpEncoder
from labs.shared.image import Image
from labs.shared.calibration_helpers import get_object_points

def find_aruko_chessboard(name_pattern: str, chessboard_size: Tuple[int, int], preview = False): 
    points_object = [] 
    points_image = [] 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

    images = glob.glob(name_pattern) 
    assert len(images) > 0 
    names_with_chessboard: List[str] = [] 

    # Get a sample image to determine the shape of images 
    sample_image = Image(images[0]) 
    image_size = sample_image.bw_data.shape[::-1] 

    for file in images: 
        image = Image(file) 

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) 
        board = cv2.aruco.CharucoBoard( 
            chessboard_size, 
            0.044, 
            0.034, 
            aruco_dict 
        ) 
        corners, ids, rejected_image_points = cv2.aruco.detectMarkers(image.bw_data, aruco_dict) 
        
        if len(corners) > 0: 
            names_with_chessboard.append(image.file_path.name) 
            print(f"Found cheesboard in {image.file_path.name}") 
 

            res, inter_corners, inter_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image.bw_data, board) 
            
            if preview:
                debug_image = copy.deepcopy(image.rgb_data) 
                cv2.aruco.drawDetectedCornersCharuco(debug_image, inter_corners, inter_ids) 
                new_size1 = int(image_size[0]/2) 
                new_size2 = int(image_size[1]/2) 
                debug_image = cv2.resize(debug_image, (new_size1, new_size2)) 
                cv2.imshow("test", debug_image) 
                cv2.waitKey() 
    
    names_with_chessboard.sort()
    return names_with_chessboard

def calculate_fov(smtx, imgSize):
    fx = smtx[0][0]
    fy = smtx[1][1]
    width, height = imgSize

    fovW = 2 * numpy.arctan(width / (2 * fx)) * 180 / numpy.pi
    fovH = 2 * numpy.arctan(height / (2 * fy)) * 180 / numpy.pi

    return fovW, fovH

def id_in_whitelist(ids_whitelist: List[str], name: str) -> bool:
    found = False

    if len(ids_whitelist) == 0:
        return True
    
    for id in ids_whitelist:
        if id in name:
            found = True
    return found

def calibrate_stereo_and_save(out_file_name: str,
                              dataset_left_name_pattern: str,
                              dataset_right_name_pattern: str,
                              board_size: Tuple[int, int],
                              square_marker_size: Tuple[float, float],
                              image_ids_whitelist: List[str]):
    points_object = []
    points_image_left = []
    points_image_right = []
    charuco_img_points_L = []
    charuco_img_points_R = []
    charuco_img_ids_L = []
    charuco_img_ids_R = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = numpy.zeros(((board_size[0]-1) * (board_size[1]-1), 3), numpy.float32)                                 # Define [70][3] array filled with zeros
    objp[:,:2] = numpy.mgrid[0:(board_size[0]-1), 0:(board_size[1]-1)].T.reshape(-1, 2) * square_marker_size[0]        # Fills indexies to elements

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50) 
    board = cv2.aruco.CharucoBoard( 
        board_size, 
        square_marker_size[0], # square length [cm]
        square_marker_size[1], # marker length [cm]
        aruco_dict 
    ) 
    detector = cv2.aruco.CharucoDetector(board)

    image_file_names_left = glob.glob(dataset_left_name_pattern)
    image_file_names_right = glob.glob(dataset_right_name_pattern)

    # Get a sample image to determine the shape of images
    sample_image = Image(image_file_names_left[0])
    image_size = sample_image.bw_data.shape[::-1]

    # image_count = 0
    # ret_l = 100
    # mtx_l = dist_l = rvecs_l = tvecs_l = None
    # ret_r = 100
    # mtx_r = dist_r = rvecs_r = tvecs_r = None
    mtx_l = dist_l = mtx_r = dist_r = None
    for name_left, name_right in zip(image_file_names_left, image_file_names_right):
        if not id_in_whitelist(image_ids_whitelist, name_left):
            continue
        # Get the images
        image_left = Image(name_left)
        image_right = Image(name_right)

        # Find patterns on LEFT
        corners_L, ids, rejected_image_points = cv2.aruco.detectMarkers(image_left.bw_data, aruco_dict) 
        if len(corners_L) > 5: 
            res_L, inter_corners_L, inter_ids_L = cv2.aruco.interpolateCornersCharuco(corners_L, ids, image_left.bw_data, board) 
            charuco_img_points_L.append(inter_corners_L)
            charuco_img_ids_L.append(inter_ids_L)
            obj_points_L, img_points_L = cv2.aruco.getBoardObjectAndImagePoints(board, inter_corners_L, inter_ids_L) # type: ignore
            detector_corners_L, detector_ids_L, _, _ = detector.detectBoard(image_left.bw_data)
        else:
            print(f"ERROR: could not find board in image {image_left.file_path.name}")
            continue
        
        # Find patterns on RIGHT
        corners_R, ids, rejected_image_points = cv2.aruco.detectMarkers(image_right.bw_data, aruco_dict) 
        if len(corners_R) > 5: 
            res_R, inter_corners_R, inter_ids_R = cv2.aruco.interpolateCornersCharuco(corners_R, ids, image_right.bw_data, board) 
            charuco_img_points_R.append(inter_corners_R)
            charuco_img_ids_R.append(inter_ids_R)
            obj_points_R, img_points_R = cv2.aruco.getBoardObjectAndImagePoints(board, inter_corners_R, inter_ids_R) # type: ignore
            detector_corners_R, detector_ids_R, _, _ = detector.detectBoard(image_right.bw_data)
        else:
            print(f"ERROR: could not find board in image {image_right.file_path.name}")
            continue

        # Collect points
        points_object.append(objp)
        points_image_left.append(detector_corners_L)
        points_image_right.append(detector_corners_R)

    # Calibrate using all collected points
    print("Calibrating...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        points_object, points_image_left, image_size, mtx_l, dist_l) # type: ignore
    print(ret_l)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        points_object, points_image_right, image_size, mtx_r, dist_r) # type: ignore
    print(ret_r)
    # ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.aruco.calibrateCameraCharuco(
    #     charuco_img_points_L, charuco_img_ids_L, board, image_size, mtx_l, dist_l) # type: ignore
    # print(ret_l)
    # ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.aruco.calibrateCameraCharuco(
    #     charuco_img_points_R, charuco_img_ids_R, board, image_size, mtx_r, dist_r) # type: ignore
    # print(ret_r)

    ret_stereo, camera_mat_left, distortion_left,\
        camera_mat_right, distortion_right, rotation_mat,\
            translation_vec, essential_mat, fundamental_mat = cv2.stereoCalibrate(points_object,
                                                                                  points_image_left,
                                                                                  points_image_right,
                                                                                  mtx_l, dist_l,
                                                                                  mtx_r, dist_r,
                                                                                  image_size,
                                                                                  criteria=criteria,
                                                                                  flags=cv2.CALIB_FIX_INTRINSIC)
    baseline_cm = numpy.linalg.norm(translation_vec) * 0.1
    print(f"Baseline in cm: {baseline_cm}")
    
    print(f"Reprojection err: {ret_stereo}")

    print(f"FOV of left camera: {calculate_fov(mtx_l, image_size)}")
    print(f"FOV of right camera: {calculate_fov(mtx_r, image_size)}")
    
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

def load_calibration_and_draw_epi(in_file_name: str,
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
            image_size, rotation_mat, translation_vec, alpha=1
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

        cv2.imwrite(f'{image_right.file_path.stem}_epi_charuco_joined.png', images_together)

def add_lines(image):
    for line in range(0, image.shape[1], 50):
        cv2.line(image, (0, line), (image.shape[1], line), (0, 0, 255), 2)

def test_chruco_stereo():

    #find_aruko_chessboard(".\\aruko_test.jpg", (8, 11), True)

    calibrate_stereo_and_save("stereo_charuco1.json",
                              ".\\Kolos_test\\left_*.png",
                              ".\\Kolos_test\\right_*.png",
                              (8, 11), (44, 34), [])
    
    load_calibration_and_draw_epi("stereo_charuco1.json",
                              ".\\Kolos_test\\left_*.png",
                              ".\\Kolos_test\\right_*.png")