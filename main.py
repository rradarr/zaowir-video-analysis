import cv2
import labs.Lab1.mono as mono
import labs.Lab2.stereo as stereo
import labs.Lab3.disparity as disparity
import labs.Lab4.motion as motion
import labs.Test.charuco_stereo as charuco_stereo
import labs.Bonus_RRO.rro as rro
import labs.Exam.depth_exam as exam

if __name__ == "__main__":
    cv2.setNumThreads(12)
    #mono.test_mono()
    #stereo.test_stereo()
    #charuco_stereo.test_chruco_stereo()
    #disparity.test_disparity()
    # disparity.test_ply()
    #motion.test_movement_detection()
    #rro.test_rro()
    exam.exam_initial()