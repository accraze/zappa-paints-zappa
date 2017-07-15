import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import os

from . import utils


def split_image(img):
    # collect all positions in the image
    xs = []

    # collect colors for each of these positions
    ys = []

    # loop over the image
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # store the inputs
            xs.append([row_i, col_i])
            # outputs that network needs to learn to predict
            ys.append(img[row_i, col_i])

    # convert lists to arrays
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def get_album_covers():
    dirname = os.getcwd() + '/covers'

    filenames = [os.path.join(dirname, fname)
             for fname in os.listdir(dirname)]

    filenames = filenames[:100]

    imgs = [plt.imread(fname)[..., :3] for fname in filenames]

    # Crop every image to a square
    imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs]

    # Then resize the square image to 100 x 100 pixels
    imgs = [resize(img_i, (100, 100)) for img_i in imgs]

    # Finally make our list of 3-D images a 4-D array with the first dimension the number of images:
    return np.array(imgs).astype(np.float32)


def sort_convolved_idx_op(convolved):
    # First flatten our convolved images so instead of many 3d images,
    # we have many 1d vectors.
    # This should convert our 4d representation of N x H x W x C to a
    # 2d representation of N x (H*W*C)
    flattened = tf.reshape(convolved, [100, 10000])
    assert(flattened.get_shape().as_list() == [100, 10000])
    # Now calculate some statistics about each of our images
    values = tf.reduce_sum(flattened, axis=1)

    # Then create another operation which sorts those values
    # and then calculate the result:
    idxs_op = tf.nn.top_k(values, k=100)[1]
    return idxs_op