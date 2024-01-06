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
import plyfile

from labs.shared.encoder import NpEncoder
from labs.shared.image import Image
from labs.shared.calibration_helpers import get_object_points

def test_rro():
    ply_data = plyfile.PlyData.read("scan.ply")

    points = numpy.array([(vertex['x'], vertex['y'], vertex['z']) for vertex in ply_data['vertex']])
    colors = numpy.array([(vertex['red'], vertex['green'], vertex['blue']) for vertex in ply_data['vertex']])
    min_bounds = numpy.min(points, axis=0)
    max_bounds = numpy.max(points, axis=0)
    scale = max(max_bounds - min_bounds)
    normalized_points = (points - min_bounds) / scale
    with open("scan.pbrt", 'w') as file:
        for point, color in zip(normalized_points, colors):
            color = [c/255.0 for c in color]
            file.write('AttributeBegin\n')
            file.write(f'Material "diffuse" "rgb reflectance" [{color[0]} {color[1]} {color[2]}]\n')
            file.write(f'Translate {point[0]} {point[1]} {point[2]}\n')
            file.write(f'Shape "sphere" "float radius" [{0.001}]\n')
            file.write('AttributeEnd\n')
    return
