class AppConfig:
    STRING_AMOUNT = 1000
    ERROR_RES = 0.001
    IS_UBUNTU = False


class EnumsConfig:
    X = 0
    Y = 1
    DEGREE = 2
    OBJECT_ID = 0
    OBJECT = 1


class ScreenConfig:
    RADIUS = 300
    DIAMETER = RADIUS * 2
    NAIL_RESOLUTION = 360  # make it an even number or everything will fuck up and I am to lazy to fix it
    MAX_NAIL_RADIUS = 5


class ImageConfig:
    IMAGE_PATH = './Test_Images/line.png'
    SHOW_SINOGRAM = True
    IMAGE_FILTER_THRESHOLD = 0  # grayscale range is [0, 1]
    ROTATE_CONST = 90
    BLOCK_DEFAULT = False


# ConstantsManagement class
class ConstantsManagement:
    def __init__(self):
        # Set constants from separate classes as attributes
        for cls in [AppConfig, ScreenConfig, EnumsConfig, ImageConfig]:
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")


# Create an instance of ConstantsManagement
Consts = ConstantsManagement()
if __name__ == "__main__":
    from colorama import Fore

    print(Fore.RED + "Stop running constants.py!!!")
