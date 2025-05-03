import numpy as np
import math

from constants import Consts
from Image_Processing import ImageProcessor
from Screen_Handler import Screen


# TODO: finish initializing the generator. Check if the changes to Screen_Handler are ok (just changed INIT). do it!

class Generator:
    def __init__(self):
        self.image = ImageProcessor()
        self.screen = Screen()
        self.radius = Consts.RADIUS
        self.center_x = Consts.RADIUS
        self.center_y = Consts.RADIUS
        self.points = []
        points = np.linspace(0, 360, Consts.NAIL_RESOLUTION)
        for point in points:
            x = self.radius * math.sin(ImageProcessor.deg2rad(point)) + self.center_x
            y = self.radius * math.cos(ImageProcessor.deg2rad(point)) + self.center_y
            self.points.append((x, y, point))

