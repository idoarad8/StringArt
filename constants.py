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
    DIAMETER = RADIUS*2
    NAIL_RESOLUTION = 500
    NAIL_RADIUS = 1


class ImageConfig:
    IMAGE_PATH = './Test_Images/Whiter.jpg'
    SHOW_SINOGRAM = True
    IMAGE_FILTER_THRESHOLD = 0 #grayscale range is [0, 1]
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