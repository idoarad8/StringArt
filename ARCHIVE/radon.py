# graphics imports
from graphics import *
# constants import
from constants import Consts
# image and graph imports
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import radon, rescale
from skimage.draw import line
# General import
import numpy as np
import math
import time


class StringsCircle:
    def __init__(self, points_res=Consts.RESOLUTION, circle_radius=Consts.RADIUS):
        self.radius = circle_radius
        self.center_x = Consts.DIAMETER / 2
        self.center_y = Consts.DIAMETER / 2
        self.last_point = 0
        self.Points = []
        self.res = points_res
        self.line_width = 2 * math.pi * self.radius / self.res
        jump = 2 * math.pi / points_res
        for index in range(points_res):
            x = circle_radius * math.sin(jump * index) + self.center_x
            y = circle_radius * math.cos(jump * index) + self.center_y
            self.Points.append((x, y, jump * index))


class Screen:
    def __init__(self, width=Consts.DIAMETER, height=Consts.DIAMETER):
        height = height or Consts.HEIGHT
        self.win = GraphWin("String Arts", width, height)
        self.win.setCoords(0, 0, width, height)
        self.objects = []
        self.id_counter = 0

    def draw_circle(self, x=0, y=0, radius=None, circle=None, circle_id=None):
        if circle_id is None:
            circle_id = str(self.id_counter)
            self.id_counter += 1
        if circle is not None:
            circ = Circle(Point(circle.x, circle.y), circle.radius)
        else:
            circ = Circle(Point(x, y), radius)
        self.objects.append((circle_id, circ))
        circ.draw(self.win)

    def draw_line(self, x0, y0, x1, y1, line_id=None):
        if line_id is None:
            line_id = str(self.id_counter)
            self.id_counter += 1
        line_objects = Line(Point(x0, y0), Point(x1, y1))
        self.objects.append((id, line_id))
        line_objects.draw(self.win)

    def delete(self, obj_id):
        for i in self.objects:
            if obj_id == i[0]:
                i[1].undraw()

    def wait_mouse(self):
        self.win.getMouse()

    def close(self):
        self.win.close()


class App:

    def __init__(self):
        self.image = None
        self.image_sinogram = None
        self.main_circle = StringsCircle()
        self.screen_handler = None
        self.id_counter = 0
        self.drawn_lines = {}
        self.initialize()

    # app functions
    def initialize(self):
        self.load_image()
        # initialize string screen
        self.screen_handler = Screen()
        self.draw_nails()
        if Consts.IS_UBUNTU:
            plt.switch_backend('WebAgg')

    def load_image(self, path=Consts.IMAGE_PATH):
        self.image = io.imread(path, True)
        self.mask_and_fit_image()
        # calculate the radon transform of the image
        self.image_sinogram = self.radon(self.image, Consts.SHOW_SINOGRAM)

    def run(self, string_amount=Consts.STRING_AMOUNT):
        alpha_max_un_normalized, r_max, max_sinogram = self.max_sinogram_point()
        while string_amount > 0 and max_sinogram > 0:
            alpha_max_degrees = self.normalize_alpha_to_degrees(alpha_max_un_normalized)
            max_point1_angle, max_point2_angle = self.sinogram_point_to_line(alpha_max_degrees, r_max)
            p1, p2 = self.find_closest_point(max_point1_angle), self.find_closest_point(max_point2_angle)
            if (p1, p2) in self.drawn_lines:
                self.image_sinogram[r_max, alpha_max_un_normalized] = 0
            else:
                self.drawn_lines[(p1, p2)] = True
                self.drawn_lines[(p2, p1)] = True
                self.draw_string(p1, p2)
                line_sinogram = self.line_radon_approx(alpha_max_un_normalized, r_max)
                self.image_sinogram = self.image_sinogram - line_sinogram
                self.image_sinogram[self.image_sinogram < 0] = 0
                string_amount -= 1
            # self.show_image(self.image_sinogram, block=True)
            # print(self.image_sinogram[r_max,alpha_max_un_normalized])
            alpha_max_un_normalized, r_max, max_sinogram = self.max_sinogram_point()
            # if string_amount % 20 == 0:
        print("DONE")
        self.show_image(self.image_sinogram)


    def wait(self):
        self.screen_handler.wait_mouse()
        self.screen_handler.close()
        plt.show(block=True)

    # drawing functions
    def draw_string(self, point1, point2):
        p1_x = self.main_circle.Points[point1][Consts.X]
        p1_y = self.main_circle.Points[point1][Consts.Y]
        p2_x = self.main_circle.Points[point2][Consts.X]
        p2_y = self.main_circle.Points[point2][Consts.Y]
        self.screen_handler.draw_line(p1_x, p1_y, p2_x, p2_y)

    def draw_nails(self):
        for nail in self.main_circle.Points:
            self.screen_handler.draw_circle(nail[Consts.X], nail[Consts.Y], Consts.NAIL_RADIUS)

    # image functions: image slicing, altering and viewing
    def mask_and_fit_image(self):
        # calculate image proportions
        nrows, ncols = self.image.shape
        row, col = np.ogrid[:nrows, :ncols]
        cnt_row, cnt_col = nrows / 2, ncols / 2
        smaller_axes = cnt_row if nrows < ncols else cnt_col
        self.image = (self.image > np.mean(self.image))

        # mask image into a circle
        outer_disk_mask = ((row - cnt_row) ** 2 + (col - cnt_col) ** 2 >
                           smaller_axes ** 2)
        self.image[outer_disk_mask] = 0

        # fit image
        ratio = smaller_axes / Consts.RADIUS if smaller_axes < Consts.RADIUS else Consts.RADIUS / smaller_axes
        self.image = rescale(self.image, ratio, anti_aliasing=False)
        nrows_rescaled, ncols_rescaled = self.image.shape
        cnt_row_rescaled, cnt_col_rescaled = nrows_rescaled / 2, ncols_rescaled / 2
        if cnt_row_rescaled < cnt_col_rescaled:
            self.image = self.image[:,
                         int(cnt_col_rescaled - cnt_row_rescaled):int(cnt_col_rescaled + cnt_row_rescaled)]
        else:
            self.image = self.image[int(cnt_row_rescaled - cnt_col_rescaled):int(cnt_row_rescaled + cnt_col_rescaled),
                         :]

    def image_between_points(self, alpha1, alpha2):
        # get the values of the points
        x1 = round(self.main_circle.radius * math.sin(alpha1) + self.main_circle.center_x)
        y1 = round(self.main_circle.radius * math.cos(alpha1) + self.main_circle.center_y)
        x2 = round(self.main_circle.radius * math.sin(alpha2) + self.main_circle.center_x)
        y2 = round(self.main_circle.radius * math.cos(alpha2) + self.main_circle.center_y)
        # if a point is on the end of an axis it's out of bound. -1 as a workaround
        if x1 == self.main_circle.center_x + self.main_circle.radius:
            x1 -= 1
        if x2 == self.main_circle.center_x + self.main_circle.radius:
            x2 -= 1
        if y1 == self.main_circle.center_y + self.main_circle.radius:
            y1 -= 1
        if y2 == self.main_circle.center_y + self.main_circle.radius:
            y2 -= 1

        line_rows, line_cols = line(x1, y1, x2, y2)
        line_mask = np.ones(self.image.shape, dtype=np.bool)
        line_mask[line_rows, line_cols] = 0
        image_line = self.image.copy()
        image_line[line_mask] = 0
        return image_line

    @staticmethod
    def show_image(img, title=None, transposed=False, block=False):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        if title is not None:
            ax.set_title(title)
        if transposed:
            ax.imshow(img.T, cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        plt.show(block=block)

    # image functions: calculating
    @staticmethod
    def radon(im, print_sinogram=False):
        # calculate the radon transform of the image
        theta = np.linspace(0.0, 180.0, max(im.shape), endpoint=False)
        sinogram = radon(im, theta=theta)
        if print_sinogram:
            dx, dy = 0.5 * 180.0 / max(im.shape), 0.5 / sinogram.shape[0]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
            ax1.set_title("Original")
            ax1.imshow(im, cmap=plt.cm.Greys_r)
            ax2.set_title("Radon transform\n(Sinogram)")
            ax2.set_xlabel("Projection angle (deg)")
            ax2.set_ylabel("Projection position (pixels)")
            ax2.imshow(
                sinogram,
                cmap=plt.cm.Greys_r,
                extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
                aspect='auto',
            )
            plt.show(block=False)
        return sinogram

    def line_radon_approx(self, alpha0, r0):
        nrows, ncols = self.image_sinogram.shape
        # row, col = np.ogrid[:nrows, :ncols]
        radius = self.main_circle.radius
        r0_norm = r0 - radius
        # sinogram_mask = self.normalize_alpha_to_radians(col - alpha0) != 0 and (((row-radius)**2 + r0_norm**2 - 2*(row-radius)*r0_norm*math.cos(self.normalize_alpha_to_radians(col - alpha0)))/math.sin(self.normalize_alpha_to_radians(col - alpha0))**2).all() <= radius**2
        # self.show_image(sinogram_mask)
        # TODO improve this. finish!

        line_sinogram = self.image_sinogram.copy()
        line_sinogram[:, :] = 0
        for r in range(nrows):
            for alpha in range(ncols):
                delta_alpha = self.normalize_alpha_to_radians(alpha - alpha0)
                r_norm = r - radius
                if delta_alpha != 0 and (
                        r_norm ** 2 +            r0_norm ** 2 - 2 * r_norm * r0_norm * math.cos(delta_alpha)) / math.sin(
                    delta_alpha) ** 2 <= radius ** 2:
                    line_sinogram[r, alpha] = 1 / abs(math.sin(delta_alpha))
                else:
                    line_sinogram[r, alpha] = 0
        line_sinogram[r0, alpha0] = radius
        # if r0 > 0:
        #     line_sinogram[r0 - 1, alpha0] = radius
        # if r0 < nrows - 1:
        #     line_sinogram[r0 + 1, alpha0] = radius
        return line_sinogram

    def line_to_sinogram_point(self, p1, p2):
        if p1 == p2:
            return None
        alpha1 = self.main_circle.Points[p1][Consts.DEGREE]
        alpha2 = self.main_circle.Points[p1][Consts.DEGREE]
        alpha_radians = (alpha1 + alpha2) / 2
        alpha_degrees = alpha_radians * 180 / math.pi
        r = math.cos((alpha2 - alpha1 / 2)) * self.main_circle.radius
        return alpha_degrees, r

    def sinogram_point_to_line(self, alpha_degrees, r):
        alpha_radians = self.deg2rad(alpha_degrees)
        radius = self.main_circle.radius
        alpha1 = alpha_radians + math.pi / 2 - math.acos((r - radius) / radius)
        alpha2 = alpha_radians + math.pi / 2 + math.acos((r - radius) / radius)
        if alpha1 == alpha2:
            alpha2 += math.pi
        return alpha1, alpha2

    def max_sinogram_point(self):
        nrows, ncols = self.image_sinogram.shape
        radius = self.main_circle.radius
        max_sinogram = -1
        r_max = -1
        alpha_max = -1
        for r in range(1, nrows):
            for alpha in range(ncols):
                weighted_sinogram_val = self.image_sinogram[r, alpha] / math.sqrt(radius ** 2 - (r - radius) ** 2)
                if weighted_sinogram_val > max_sinogram or max_sinogram == -1:
                    max_sinogram = weighted_sinogram_val
                    r_max = r
                    alpha_max = alpha
        return alpha_max, r_max, max_sinogram

    def find_closest_point(self, alpha=None, p1=None):  # alpha -> angle in radians. p1 -> tuple of x and y (x,y)
        min_err = -1
        min_index = -1
        for index, point in enumerate(self.main_circle.Points):
            if alpha is not None:
                err = abs(alpha - point[Consts.DEGREE])
            elif p1 is not None:
                err = math.sqrt((p1[Consts.X] - point[Consts.X]) ** 2 + (p1[Consts.Y] - point[Consts.Y]) ** 2)
            else:
                return None
            if min_err == -1 or min_err > err:
                min_err = err
                min_index = index
        return min_index

    def normalize_alpha_to_degrees(self, alpha):
        return alpha * 180 / (self.main_circle.radius * 2)

    def normalize_alpha_to_radians(self, alpha):
        return alpha * math.pi / (self.main_circle.radius * 2)

    @staticmethod
    def deg2rad(alpha):
        return alpha * math.pi / 180

    @staticmethod
    def rad2deg(alpha):
        return alpha * 180 / math.pi


def main():
    app_handler = App()
    app_handler.run()
    app_handler.wait()


if __name__ == "__main__":
    main()
