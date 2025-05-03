import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import radon, rescale, rotate
import numpy as np
import math

from constants import Consts


# TODO: fix the convertion between sinogram and line (both ways). make sure i understand the radon angles and how they compare to the image


class ImageProcessor:

    def __init__(self, path=Consts.IMAGE_PATH):
        self.image = io.imread(path, as_gray=True)
        self.mask_and_fit_image()
        self.sinogram = self.radon(rotate(self.image, Consts.ROTATE_CONST), Consts.SHOW_SINOGRAM,
                                   theta=np.linspace(0, 180, int(Consts.NAIL_RESOLUTION / 2)))
        if Consts.IS_UBUNTU:
            plt.switch_backend('WebAgg')

    def mask_and_fit_image(self, threshold=Consts.IMAGE_FILTER_THRESHOLD):
        # calculate image proportions
        nrows, ncols = self.image.shape
        row, col = np.ogrid[:nrows, :ncols]
        cnt_row, cnt_col = nrows / 2, ncols / 2
        smaller_axes = cnt_row if nrows < ncols else cnt_col
        self.image[self.image < threshold] = 0

        # mask image into a circle
        outer_disk_mask = ((row - cnt_row) ** 2 + (col - cnt_col) ** 2 >
                           smaller_axes ** 2)
        self.image[outer_disk_mask] = 0

        # fit image to radius (change radius in Constants file
        ratio = Consts.RADIUS / smaller_axes
        self.image = rescale(self.image, ratio, anti_aliasing=True)
        nrows_rescaled, ncols_rescaled = self.image.shape
        cnt_row_rescaled, cnt_col_rescaled = nrows_rescaled / 2, ncols_rescaled / 2
        if cnt_row_rescaled < cnt_col_rescaled:
            self.image = self.image[:,
                         int(cnt_col_rescaled - cnt_row_rescaled):int(cnt_col_rescaled + cnt_row_rescaled)]
        else:
            self.image = self.image[int(cnt_row_rescaled - cnt_col_rescaled):int(cnt_row_rescaled + cnt_col_rescaled),
                         :]

    def line_radon_approx(self, alpha0, r0):
        radius = Consts.RADIUS
        r0_norm = r0 - radius

        nrows, ncols = self.sinogram.shape
        r, alpha = np.ogrid[:nrows, :ncols]
        delta_alpha_radians = self.deg2rad(alpha - alpha0)
        delta_alpha_radians[delta_alpha_radians == 0] = Consts.ERROR_RES ** 2
        r_norm = r - Consts.RADIUS

        line_sinogram = np.zeros(self.sinogram.shape)
        line_sinogram[:, :] = 1 / abs(np.sin(delta_alpha_radians))
        line_sinogram[r0, alpha0] = nrows
        line_sinogram[(r_norm ** 2 + r0_norm ** 2 - 2 * r_norm * r0_norm * np.cos(delta_alpha_radians)) / np.sin(
            delta_alpha_radians) ** 2 >= radius ** 2] = 0
        return line_sinogram

    def show(self, block=Consts.BLOCK_DEFAULT):
        self.show_image(self.image, "The Image", block=block)

    def show_sinogram(self, block=Consts.BLOCK_DEFAULT, normalized=False):
        if normalized:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
            ax1.set_title("Normalized Sinogram")
            ax1.set_xlabel("Projection angle (deg)")
            ax1.set_ylabel("Projection position (pixels)")
            ax1.imshow(self.sinogram, aspect='auto', extent=(0, 180, Consts.DIAMETER, 0), cmap='gray')
            plt.show(block=block)
        else:
            self.show_image(self.sinogram, "Image Sinogram", block=block)

    @staticmethod
    def line_to_sinogram_point(angles_in_radians=None, points_coords=None):
        if points_coords is None and angles_in_radians is None:
            return None
        elif points_coords is not None and angles_in_radians is None:
            x1 = points_coords[0][Consts.X]
            y1 = points_coords[0][Consts.Y]
            x2 = points_coords[1][Consts.X]
            y2 = points_coords[1][Consts.Y]
            alpha1 = math.asin(y1 / math.sqrt(x1 ** 2 + y1 ** 2))
            alpha2 = math.asin(y2 / math.sqrt(x2 ** 2 + y2 ** 2))
        else:
            alpha1 = angles_in_radians[0]
            alpha2 = angles_in_radians[1]
        if alpha1 == alpha2:
            return None
        alpha_radians = (alpha1 + alpha2) / 2
        alpha_degrees = ImageProcessor.rad2deg(alpha_radians)
        r = math.cos((alpha2 - alpha1) / 2) * Consts.RADIUS
        return alpha_degrees, r

    @staticmethod
    def sinogram_point_to_line(alpha_degrees, r):
        alpha_radians = ImageProcessor.deg2rad(alpha_degrees)
        radius = Consts.RADIUS
        alpha1 = alpha_radians + math.pi / 2 - math.acos((r - radius) / radius)
        alpha2 = alpha_radians + math.pi / 2 + math.acos((r - radius) / radius)
        if alpha1 == alpha2:
            alpha2 += math.pi
        return alpha1, alpha2

    @staticmethod
    def radon(im, print_sinogram=False, block=Consts.BLOCK_DEFAULT, theta=None):
        # The angles that we care about in this program are equally spaced on a circle
        # and so half of them are equally spaced on a 180 degree arc
        sinogram = radon(im, theta=theta, circle=True)
        if print_sinogram:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
            ax1.set_title("Original")
            ax1.imshow(im, cmap='gray')
            ax2.set_title("Radon transform\n(Sinogram)")
            ax2.set_xlabel("Projection angle (deg)")
            ax2.set_ylabel("Projection position (pixels)")
            ax2.imshow(sinogram, aspect='auto', cmap='gray')
            plt.show(block=block)
        return sinogram

    @staticmethod
    def show_image(img, title=None, block=Consts.BLOCK_DEFAULT):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        if title is not None:
            ax.set_title(title)
        ax.imshow(img, cmap='gray')
        plt.show(block=block)

    @staticmethod
    def deg2rad(alpha):
        return alpha * math.pi / 180

    @staticmethod
    def rad2deg(alpha):
        return alpha * 180 / math.pi

    @staticmethod
    def normalize_sinogram_to_degrees(alpha_un_normalized):
        return alpha_un_normalized * 180 / int(Consts.NAIL_RESOLUTION / 2)

    @staticmethod
    def normalize_sinogram_to_radians(alpha_un_normalized):
        return alpha_un_normalized * math.pi / int(Consts.NAIL_RESOLUTION / 2)
