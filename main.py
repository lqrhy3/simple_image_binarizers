import os
import numpy as np
import cv2
from tqdm import tqdm
from functools import partial
from utils import read_grayscale_image, save_grayscale_image, plot_hist
from algorithms import binarize_otsu, binarize_modified_otsu, binarize_niblack, binarize_modidfied_niblack
from matplotlib import pyplot as plt

binarizers = {
    'otsu': binarize_otsu,
    'modified_otsu': binarize_modified_otsu,
    # 'niblack_41x41_alpha=1.2': partial(binarize_niblack, kernel_size=41, alpha=1.2),
    # 'modified_niblack_21x21_alpha=1.2': partial(binarize_modidfied_niblack, kernel_size=41, alpha=1.2, std_min=8),
}


def main(resize: bool = False, hist: bool = False, make_single_image: bool = False):
    abs_path = os.path.abspath(os.path.dirname(__file__))
    input_dir = os.path.join(abs_path, 'images', 'input')
    output_dir = os.path.join(abs_path, 'images', 'output')

    imnames = sorted(os.listdir(input_dir), key=lambda key: int(os.path.splitext(key)[0]))
    for imname in tqdm(imnames, desc='Loop over all images', position=0, leave=True):
        image = read_grayscale_image(os.path.join(input_dir, imname))
        if resize:
            h, w = image.shape
            scale = 720 / max(h, w)
            h, w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (w, h))

        if make_single_image:
            output_image = []
        else:
            output_image = None

        for i, (name, binarize_fn) in enumerate(tqdm(binarizers.items(), desc=f' Processing image {imname}', position=1, leave=False)):
            binarized_image, thr = binarize_fn(image)
            binarized_image = np.concatenate([
                np.zeros((40, binarized_image.shape[1]), dtype=np.uint8), binarized_image]
            )

            binarized_image = cv2.putText(
                binarized_image,
                f'{name}',
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 1, cv2.LINE_AA
            )
            if make_single_image:
                output_image.append(binarized_image)
            else:
                output_image = binarized_image
                save_grayscale_image(
                    output_image,
                    os.path.join(output_dir, os.path.splitext(imname)[0] + '_' + name + os.path.splitext(imname)[1])
                )

            if hist and thr is not None:
                plot_hist(image, thr, title=imname, show=True, color='r' if i == 0 else 'g')

        if make_single_image:
            output_image = np.concatenate(output_image, axis=1)
            save_grayscale_image(output_image, os.path.join(output_dir, imname))


if __name__ == '__main__':
    main(resize=True, hist=False, make_single_image=False)
