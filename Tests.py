from difflib import restore

import numpy as np

from Screen_Handler import Screen
from Image_Processing import ImageProcessor
from skimage.transform import iradon
from Generator import Generator
from constants import Consts

from skimage.draw import line


def screen_test():
    screen = Screen()
    screen.draw_line(0, 0, 100, 100)
    screen.draw_circle(100, 100, 15)
    screen.wait_mouse()
    screen.draw_line(100, 100, 200, 200)
    screen.wait_mouse()
    screen.close()


def approx_radon_and_convertions_test(angle=160, r=50):
    image = ImageProcessor()
    if 10 < r < (Consts.DIAMETER - 10) and 10 < angle < (Consts.NAIL_RESOLUTION / 2 - 10):
        print(f'angle:{angle}, r:{r}')
        line_radon = image.line_radon_approx(angle, r)
        ImageProcessor.show_image(line_radon, "radon line approx")
        ImageProcessor.show_image(line_radon[r - 10:r + 10, angle - 10:angle + 10], "radon line approx (Cool Area)")
        alpha1, alpha2 = ImageProcessor.sinogram_point_to_line(angle, r)
        print(f'alpha1:{ImageProcessor.rad2deg(alpha1)} alpha2:{ImageProcessor.rad2deg(alpha2)}')
        restored_angle, restored_r = ImageProcessor.line_to_sinogram_point(angles_in_radians=(alpha2, alpha1))
        print(f'restored angle:{restored_angle}, restored r:{restored_r}')
    else:
        print("r and angle should have 10 gap from edge")


def image_show_and_radon_test():
    image = ImageProcessor()
    image.show()
    image.show_sinogram(degrees_correct=True)
    image.show_sinogram(degrees_correct=False)
    ImageProcessor.show_image(image.line_radon_approx(45,300))
    ImageProcessor.show_image(iradon(image.line_radon_approx(45,300)), "is good?")
    ImageProcessor.show_image(iradon(image.sinogram), "inverse of radon")

def line_image_test():
    image = ImageProcessor()
    line = iradon(image.line_radon_approx(45,300))
    image_new = ImageProcessor(line)

def test_generator():
    gen = Generator()
    gen.show_normalized_masked_sinogram()
    gen.run()


if __name__ == "__main__":
    test_generator()

    # test_generator()
    screen_test()
