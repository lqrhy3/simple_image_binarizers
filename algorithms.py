import numpy as np
import scipy.signal

from typing import Tuple
import numpy.typing as npt
NDArray = npt.NDArray


def binarize_otsu(image: NDArray):
    assert image.dtype == np.uint8

    pixels = image.flatten()
    thresholds = np.arange(np.min(image) + 0.1, np.max(image) - 0.1, 1)

    min_criterion, best_thr = np.inf, 0
    for thr in thresholds:
        bg_pixels = pixels[pixels < thr]
        w_bg = len(bg_pixels) / len(pixels)
        var_bg = np.var(bg_pixels)

        fg_pixels = pixels[pixels >= thr]
        w_fg = len(fg_pixels) / len(pixels)
        var_fg = np.var(fg_pixels)

        criterion = w_bg * var_bg + w_fg * var_fg  # within class variance
        if criterion < min_criterion:
            min_criterion = criterion
            best_thr = thr

    image = ((image > best_thr) * 255).astype(np.uint8)

    return image, best_thr


def binarize_modified_otsu(image: NDArray):
    assert image.dtype == np.uint8

    pixels = image.flatten()
    thresholds = np.arange(np.min(image) + 0.1, np.max(image) - 0.1, 1)

    max_criterion, best_thr = -np.inf, 0
    for thr in thresholds:
        bg_pixels = pixels[pixels < thr]
        w_bg = len(bg_pixels) / len(pixels)
        var_bg = np.var(bg_pixels)

        fg_pixels = pixels[pixels >= thr]
        w_fg = len(fg_pixels) / len(pixels)
        var_fg = np.var(fg_pixels)

        criterion = w_bg * np.log2(w_bg) + w_fg * np.log2(w_fg) - np.log2(w_bg * var_bg + w_fg * var_fg)
        if criterion > max_criterion:
            max_criterion = criterion
            best_thr = thr

    image = ((image > best_thr) * 255).astype(np.uint8)

    return image, best_thr


def binarize_niblack(image: NDArray, kernel_size: int or Tuple[int, int], alpha: float = -0.2) -> NDArray:
    assert image.dtype == np.uint8

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    assert all([ks % 2 == 1 for ks in kernel_size])

    image = image.astype(float)
    mean_kernel = np.ones((kernel_size[0], kernel_size[1])) / (kernel_size[0] * kernel_size[1])
    mean = scipy.signal.convolve2d(image, mean_kernel, mode='same', boundary='symm')
    mean_of_sq = scipy.signal.convolve2d(image ** 2, mean_kernel, mode='same', boundary='symm')
    std = np.sqrt(mean_of_sq - mean ** 2)

    thr = mean - alpha * std

    image = ((image > thr) * 255).astype(np.uint8)

    return image, None


def binarize_modidfied_niblack(
        image: NDArray, kernel_size: int or Tuple[int, int], alpha: float, std_min: float
) -> NDArray:
    assert image.dtype == np.uint8

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    assert all([ks % 2 == 1 for ks in kernel_size]), 'kernel sizes must be odd.'

    image = image.astype(float)
    thr = np.empty_like(image)
    mean_ = np.zeros_like(image)
    std_ = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x_margin, y_margin = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
            binarized = False
            while not binarized:
                top, bottom = max(0, i - y_margin), i + y_margin + 1
                left, right = max(0, j - x_margin), j + x_margin + 1

                window = image[top:bottom, left:right]
                mean = np.mean(window)
                std = np.std(window)
                if std > std_min:
                    thr[i, j] = mean - alpha * std
                    mean_[i, j] = mean
                    std_[i, j] = std
                    binarized = True
                else:
                    x_margin, y_margin = x_margin * 2, y_margin * 2

                    if (
                            min(j + x_margin, image.shape[1]) - max(0, j - x_margin) > image.shape[1]
                            and
                            min(i + y_margin, image.shape[0]) - max(0, i - y_margin) > image.shape[0]
                    ):
                        return image, 0

    image = ((image > thr) * 255).astype(np.uint8)

    return image, None
