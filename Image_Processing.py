import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import radon, rescale
from skimage.draw import line
import numpy as np
import math

from constants import Consts


class ImageProcessor:

    def __init__(self, path=Consts.IMAGE_PATH):
        self.image = io.imread(path, as_gray=True)
        self.mask_and_fit_image()
        self.sinogram = self.radon(self.image, Consts.SHOW_SINOGRAM)
        # plt.switch_backend('Qt5Agg')

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
        r_norm = r-Consts.RADIUS

        line_sinogram = np.zeros(self.sinogram.shape)
        line_sinogram[:,:] = 1 / abs(np.sin(delta_alpha_radians))
        line_sinogram[r0, alpha0] = nrows
        line_sinogram[(r_norm ** 2 + r0_norm ** 2 - 2 * r_norm * r0_norm * np.cos(delta_alpha_radians)) / np.sin(delta_alpha_radians) ** 2 > radius ** 2] = 0
        return line_sinogram

    def show(self, block=Consts.BLOCK_DEFAULT):
        self.show_image(self.image, "The Image", block=block)

    def show_sinogram(self, spread=False, block=Consts.BLOCK_DEFAULT):
        if spread:
            dx, dy = 0.5 * 180.0 / max(self.image.shape), 0.5 / self.sinogram.shape[0]
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
            ax1.set_title("Radon transform Spread\n(Sinogram)")
            ax1.set_xlabel("Projection angle (deg)")
            ax1.set_ylabel("Projection position (pixels)")
            ax1.imshow(
                self.sinogram,
                cmap=plt.cm.Greys_r,
                extent=(-dx, 180.0 + dx, -dy, self.sinogram.shape[0] + dy),
                aspect='auto',
            )
            plt.show(block=block)
        else:
            self.show_image(self.sinogram, "Image Sinogram Original", block=block)

    @staticmethod
    def show_image(img, title=None, block=Consts.BLOCK_DEFAULT):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        if title is not None:
            ax.set_title(title)
        ax.imshow(img, cmap='gray')
        plt.show(block=block)

    @staticmethod
    def radon(im, print_sinogram=False, block=Consts.BLOCK_DEFAULT):
        # calculate the radon transform of the image
        sinogram = radon(im, circle=True)
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
            plt.show(block=block)
        return sinogram

    @staticmethod
    def deg2rad(alpha):
        return alpha * math.pi / 180

    @staticmethod
    def rad2deg(alpha):
        return alpha * 180 / math.pi
