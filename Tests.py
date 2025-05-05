from difflib import restore

from Screen_Handler import Screen
from Image_Processing import ImageProcessor
from skimage.transform import iradon
from Generator import Generator

from skimage.draw import line


def screen_test():
    screen = Screen()
    screen.draw_line(0, 0, 100, 100)
    screen.draw_circle(100, 100, 15)
    screen.wait_mouse()
    screen.draw_line(100, 100, 200, 200)
    screen.wait_mouse()
    screen.close()


def image_proc_test():
    image = ImageProcessor()
    image.show()
    image.show_sinogram(normalized=True)
    # image.show_sinogram(spread=True)
    angle = 100
    r = 500
    print(f'angle:{angle}, r:{r}')
    line_radon = image.line_radon_approx(angle, r)
    ImageProcessor.show_image(line_radon, "radon line approx")
    ImageProcessor.show_image(line_radon[r-10:r+10, angle-10:angle+10], "radon line approx (Cool Area)")
    alpha1, alpha2 = ImageProcessor.sinogram_point_to_line(angle, r)
    print(f'alpha1:{ImageProcessor.rad2deg(alpha1)} alpha2:{ImageProcessor.rad2deg(alpha2)}')
    restored_angle, restored_r = ImageProcessor.line_to_sinogram_point(angles_in_radians=(alpha1, alpha2))
    print(f'restored angle:{restored_angle}, restored r:{restored_r}')
    ImageProcessor.show_image(iradon(image.sinogram), "inverse of radon")


def test_generator():
    gen = Generator()


if __name__ == "__main__":
    image_proc_test()
    # test_generator()
    screen_test()
