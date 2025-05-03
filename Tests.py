from Screen_Handler import Screen
from Image_Processing import ImageProcessor
from skimage.transform import iradon


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
    image.show_sinogram()
    # image.show_sinogram(spread=True)
    ImageProcessor.show_image(image.line_radon_approx(150, 200), "radon line approx")
    ImageProcessor.show_image(image.line_radon_approx(150, 200)[190:210, 140:160], "radon line approx (Cool Area)")
    ImageProcessor.show_image(iradon(image.sinogram), "inverse of radon")


if __name__ == "__main__":
    image_proc_test()
    screen_test()
