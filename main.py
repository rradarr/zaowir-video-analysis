import cv2
import labs.Lab1.mono as mono
import labs.Lab2.stereo as stereo
import labs.Lab3.disparity as disparity
import labs.Test.charuco_stereo as charuco_stereo

if __name__ == "__main__":
    cv2.setNumThreads(12)
    #mono.test_mono()
    #stereo.test_stereo()
    charuco_stereo.test_chruco_stereo()