import numpy

def get_object_points():
    objp = numpy.zeros((8*6,3), numpy.float32)
    objp[:,:2] = numpy.mgrid[0:8,0:6].T.reshape(-1,2) * 0.0285
    return objp