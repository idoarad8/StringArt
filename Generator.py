import numpy as np
import math

from constants import Consts
from Image_Processing import ImageProcessor
from Screen_Handler import Screen


# TODO: fix run function

class Generator:
    def __init__(self):
        self.image = ImageProcessor()
        self.screen = Screen()
        self.radius = Consts.RADIUS
        self.center_x = Consts.RADIUS
        self.center_y = Consts.RADIUS
        self.string_amount = Consts.STRING_AMOUNT
        self.normalized_masked_sinogram = None
        self.valid_r_values = []
        self.points = []
        self.initialize()

    def initialize(self):
        self.set_sinogram()
        self.set_points()
        self.draw_nails()

    def set_sinogram(self):
        self.normalized_masked_sinogram = self.image.normalize_sinogram()
        self.normalized_masked_sinogram[self.image.mask_for_sinogram()] = 0
        for r in range(self.normalized_masked_sinogram.shape[0]):
            check = True
            for angle in range(self.normalized_masked_sinogram.shape[1]):
                if self.normalized_masked_sinogram[r, angle] != 0 and check:
                    check = False
                    self.valid_r_values.append(r)

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
        computed_radius = 2 * math.pi * Consts.RADIUS / Consts.NAIL_RESOLUTION
        computed_radius -= 0.9 * computed_radius
        nail_radius = min(Consts.MAX_NAIL_RADIUS, computed_radius)
        for nail in self.points:
            self.screen.draw_circle(nail[Consts.X], nail[Consts.Y], nail_radius)

    def max_normalized_sinogram_point(self, starting_point_index=None):
        if starting_point_index is None:
            max_index_flat = np.argmax(self.normalized_masked_sinogram)
            max_r = int(max_index_flat / (Consts.NAIL_RESOLUTION / 2))
            max_alpha_degree = int(max_index_flat % (Consts.NAIL_RESOLUTION / 2))
            return -1, max_r, max_alpha_degree
        else:
            starting_point = self.points[starting_point_index]
            max_ending_point = -1
            max_alpha = -1
            max_r = 0
            max_ending_point_value = -1
            for index, point in enumerate(self.points):
                if index != starting_point_index:
                    alpha1 = ImageProcessor.deg2rad(starting_point[Consts.DEGREE])
                    alpha2 = ImageProcessor.deg2rad(point[Consts.DEGREE])
                    alpha_sinogram, r_sinogram = ImageProcessor.line_to_sinogram_point(
                        angles_in_radians=(alpha1, alpha2))
                    alpha_sinogram = int(alpha_sinogram * (Consts.NAIL_RESOLUTION / 2) / 180)
                    r_sinogram = self.closest_valid_r(r_sinogram)
                    value = self.normalized_masked_sinogram[r_sinogram,alpha_sinogram]
                    if value > max_ending_point_value or max_ending_point_value == -1:
                        max_ending_point_value = value
                        max_alpha = alpha_sinogram
                        max_r = r_sinogram
                        max_ending_point = index
                return max_ending_point, max_r, max_alpha

    def closest_valid_r(self, exact_r):
        min_diff = -1
        min_index = -1
        for index, r in enumerate(self.valid_r_values):
            diff = abs(r - exact_r)
            if diff < min_diff or min_diff == -1:
                min_diff = diff
                min_index = index
        return self.valid_r_values[min_index]

    def find_closet_point_index(self, exact_alpha):
        min_diff = -1
        min_index = -1
        for index, point in enumerate(self.points):
            diff = abs(point[Consts.DEGREE] - exact_alpha)
            if diff < min_diff or min_diff == -1:
                min_diff = diff
                min_index = index
        return min_index

    def run(self):
        current_point = None
        string_amount = Consts.STRING_AMOUNT
        while string_amount > 0:
            next_point, max_r, max_alpha = self.max_normalized_sinogram_point(current_point)
            max_alpha_degree = ImageProcessor.convert_sinogram_to_degrees(max_alpha)
            if next_point == -1:
                alpha1, alpha2 = self.image.sinogram_point_to_line(max_alpha_degree, max_r)
                current_point = self.find_closet_point_index(alpha1)
                next_point = self.find_closet_point_index(alpha2)
            self.draw_string(current_point, next_point)
            self.normalized_masked_sinogram -= self.image.line_radon_approx(max_alpha, max_r)
            print(current_point,next_point)
            string_amount -= 1


        pass

    def show_normalized_masked_sinogram(self, block=Consts.BLOCK_DEFAULT):
        ImageProcessor.show_image(self.normalized_masked_sinogram, "normalized masked sinogram", block=block)
