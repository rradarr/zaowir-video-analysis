import cv2
import labs.Lab1.mono as mono
import labs.Lab2.stereo as stereo

if __name__ == "__main__":
    cv2.setNumThreads(12)
    mono.test_mono()
    #stereo.test_stereo()