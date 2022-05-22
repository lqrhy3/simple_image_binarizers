import cv2
import numpy.typing as npt
from matplotlib import pyplot as plt

NDArray = npt.NDArray


def read_grayscale_image(path: str) -> NDArray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert len(image.shape) < 3

    return image


def save_grayscale_image(image: NDArray, path_to_save: str):
    cv2.imwrite(path_to_save, image)


def plot_hist(image: NDArray, thr: int or float = None, title: str = '', color: str = 'r', show: bool = False):
    pixels = image.ravel()
    b, bins, patches = plt.hist(pixels, 255, color='blue')
    plt.title(title)
    plt.xlim([0, 255])
    plt.vlines(thr, ymin=0, ymax=max(b), color=color, linewidth=2.)
    if show:
        plt.show()

