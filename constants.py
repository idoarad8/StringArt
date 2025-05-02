class AppConfig:
    STRING_AMOUNT = 1000
    ERROR_RES = 0.001
    IS_UBUNTU = False


class EnumsConfig:
    X = 0
    Y = 1
    DEGREE = 2


class CircleConfig:
    RESOLUTION = 500
    RADIUS = 200
    DIAMETER = RADIUS*2
    NAIL_RADIUS = 1


class ImageConfig:
    IMAGE_PATH = './circle.jpg'
    SHOW_SINOGRAM = True


# ConstantsManagement class
class ConstantsManagement:
    def __init__(self):
        # Set constants from separate classes as attributes
        for cls in [AppConfig, CircleConfig, EnumsConfig, ImageConfig]:
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