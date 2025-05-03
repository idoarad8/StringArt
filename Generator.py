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
        self.initialize()

    def initialize(self):
        self.set_points()
        self.draw_nails()

    def set_points(self):
        points = np.linspace(0, 360, Consts.NAIL_RESOLUTION)
        for point in points:
            x = self.radius * math.sin(ImageProcessor.deg2rad(point)) + self.center_x
            y = self.radius * math.cos(ImageProcessor.deg2rad(point)) + self.center_y
            self.points.append((x, y, point))

    def draw_string(self, point1, point2):
        p1_x = self.points[point1][Consts.X]
        p1_y = self.points[point1][Consts.Y]
        p2_x = self.points[point2][Consts.X]
        p2_y = self.points[point2][Consts.Y]
        self.screen.draw_line(p1_x, p1_y, p2_x, p2_y)

    def draw_nails(self):
        computed_radius = 2*math.pi*Consts.RADIUS/Consts.NAIL_RESOLUTION
        computed_radius -= 0.9*computed_radius
        nail_radius = min(Consts.MAX_NAIL_RADIUS, computed_radius)
        for nail in self.points:
            self.screen.draw_circle(nail[Consts.X], nail[Consts.Y], nail_radius)

