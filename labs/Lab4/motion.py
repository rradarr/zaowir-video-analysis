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

def test_sparse_optical_flow():
    # Load image sequence
    file_names = glob.glob(".\\s3_forflow\\left_*.png")
    images = []

    for file in file_names:
        image = Image(file)
        images.append(image.bw_data)

    # parameters for ShiTomasi corner detection
    [maxCorners, qualityLevel, minDistance, blockSize] = [30, 0.3, 7, 7]

    # parameters for lucas kanade optical flow
    winSize = (700,700)
    maxLevel = 4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    # create some random colors
    color = []
    for n in range(maxCorners):
        color.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # take first frame and find corners in it
    last_points = cv2.goodFeaturesToTrack(images[0], maxCorners, qualityLevel, minDistance, blockSize=blockSize)
    current_points = None
    mask = numpy.zeros_like(images[0])
    FPS = 30
    
    
    for n, frame in enumerate(images[1:], start=1):
        # calculate optical flow
        current_points, status, error = cv2.calcOpticalFlowPyrLK(images[n-1],
                                                                frame,
                                                                last_points,
                                                                None,
                                                                winSize=winSize,
                                                                maxLevel=maxLevel,
                                                                criteria=criteria) # type: ignore
        # Select good points 
        good_new = []
        good_old = []
        for i, stat in enumerate(status):
            if stat == 1:
                good_new.append(current_points[i])
                good_old.append(last_points[i]) 

        # draw the tracks 
        for i, (new, old) in enumerate(zip(good_new, good_old)): 
            a, b = new.ravel() 
            c, d = old.ravel() 
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i], 2) 
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i], -1) 
        img = cv2.add(frame, mask) 

        last_points = numpy.array(good_new)
  
        cv2.imshow('frame', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

def test_dense_optical_flow():
    # Load image sequence
    file_names = glob.glob(".\\s3\\left_*.png")
    images: List[Image] = []

    for file in file_names:
        image = Image(file)
        images.append(image)

    # take first frame of the video
    last_frame = images[0]
    hsv = numpy.zeros_like(last_frame.rgb_data)
    hsv[...,1] = 255
    flow = None
    
    for frame in images[1:]:
        flow = cv2.calcOpticalFlowFarneback(last_frame.bw_data,frame.bw_data, flow, 0.5, 3, 10, 3, 5, 1.2, 0) # type: ignore
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/numpy.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # type: ignore
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        last_frame = frame

def test_movement_detection():
    SCALING_F = 0.5
    cap = cv2.VideoCapture(cv2.samples.findFile("motion_vid.mp4"))
    ret, last_frame = cap.read()
    last_frame = cv2.resize(last_frame, (0,0), fx=SCALING_F, fy=SCALING_F)

    # take first frame of the video
    hsv = numpy.zeros_like(last_frame)
    hsv[...,1] = 255
    flow = None
    last_frame = cv2.cvtColor(last_frame,cv2.COLOR_BGR2GRAY)
    time_sum = 0
    time_measurements = 1

    while(1):
        start = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0,0), fx=SCALING_F, fy=SCALING_F)
        col_frame = copy.deepcopy(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        flow = cv2.calcOpticalFlowFarneback(last_frame,frame, flow, 0.5, 3, 10, 3, 5, 1.2, 0) # type: ignore
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/numpy.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # type: ignore
        hsv = cv2.GaussianBlur(hsv, (31, 31), 0)
        hsv = cv2.threshold(hsv, 50, 255, cv2.THRESH_BINARY)[1]
        hsv = cv2.dilate(hsv, None, iterations=4)
        hsv = cv2.erode(hsv, None, iterations=8)

        hsv_for_cont = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        hsv_for_cont = cv2.cvtColor(hsv_for_cont,cv2.COLOR_BGR2GRAY)
        contrs,_ = cv2.findContours(hsv_for_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        temp_mask = numpy.zeros_like(hsv_for_cont)
        detections = []
        for cnt in contrs:
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            # get flow angle inside of contour
            cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)
            flow_angle = hsv[numpy.nonzero(temp_mask)]


            if (area > 10000):
                cv2.rectangle(col_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',col_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        last_frame = frame

        time_sum += time.time() - start
        time_measurements += 1

    print(f"Mean time of frame processing: {time_sum / time_measurements}s.")

def test_specific_movement():
    ...