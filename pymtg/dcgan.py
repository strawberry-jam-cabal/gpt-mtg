"""
This code was modified from the great work of Erik Linder-Noren.
Specifically Added:
- Model saving and loading to retrain existing models and generate new images
- Model auto sizing based on inputs
- Configurable model parameters
- dropout parameters for experimentation
- Data Generators for loading data from directories
- label smoothing

MIT License

Copyright (c) 2017 Erik Linder-Norén

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Some personal paths
# /Users/tetracycline/data/images/mtg/magic_cards
["/Users/tetracycline/repos/datascience/usda_water_colors"],
["/tmp/tetracycline/data/images/usda_water_colors", "/tmp/tetracycline/data/images/hunt_plants_watercolor"],
["/tmp/tetracycline/data/images/mtg/magic_cards"],
"""

# TODO:: load_all_data is confusing and is used for the watercolors because of preprocessing shenanigans
# I think this needs to be turned off for basically everything else.
# This is confusing and should be cleaned

# TODO
# The generator has good results when 32 is the number of final filters and 512 is the number of input filters
# Can we get good performance with fewer final layer filters i.e. 16.  If we can we can reduce the model size substantially
# This would allow bigger images.  If not we might be limited by the filter size.

# TODO:: ADD conv2d transpose instead of upsampling, this has been shown to have better performance
# TODO:: Add decaying noise to discriminator input
# TODO:: ADD DROPOUT WHICH RUNS DURING TRAINING AS WELL AT 50% to more layers
from functools import partial
import math
import multiprocessing
import os
from typing import List, Optional, Tuple

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import (
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    SpatialDropout2D,
    Lambda,
)
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, save_model, load_model
from keras.optimizers import Adam
import matplotlib.image as mpimg
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import sys

if "linux" in sys.platform:
    os.environ["LC_AL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
else:
    os.environ["LC_AL"] = "en_US.utf-8"
    os.environ["LANG"] = "en_US.utf-8"

import click


def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))


@click.group()
def main():
    pass


def auto_trim(im):
    """Trims the image based on the median of the four corners.

    Args:
        im:

    Returns:

    """
    im_size = im.size
    corners = np.array(
        [
            im.getpixel((0, 0)),
            im.getpixel((im_size[0] - 1, 0)),
            im.getpixel((0, im_size[1] - 1)),
            im.getpixel((im_size[0] - 1, im_size[1] - 1)),
        ]
    )
    median = tuple(np.median(corners, axis=0).astype(int))
    bg = Image.new(im.mode, im.size, median)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def image_random_data_generator(
    data_path: str,
    image_size: Tuple[int, int],
    batch_size: int = 256,
    image_crop: Optional[int] = None,
) -> Tuple[np.array, np.array]:
    """Learning generator for gans based on file paths

    This method acts as a generator where it randomly samples files from a directory.  It loads in
    batch_size of these images and resized/crops them to the desired dimensions.  Then returns a tuple
    where the first element is the batch of processed images, and the second is a vector of ones indicating
    that these are ground truth images.

    Args:
        data_path: A path to a directory full of images we want to train on
        image_size: The dimensions the image should be resized to
        batch_size: The number of images in our batch
        image_crop: How much we want to take off of all sides of the image.

    Returns:
        images:
        ones:
    """
    file_names = os.listdir(data_path)
    file_names = [file_name for file_name in file_names if ".DS_Store" not in file_name]
    y_train = np.ones([batch_size, 1])
    while True:
        # Randomly Sample batch_size worth of images
        sample_idxs = np.random.randint(0, len(file_names), batch_size)

        sampled_files = [file_names[i] for i in sample_idxs]

        imgs = []
        for file_name in sampled_files:
            img = mpimg.imread(os.path.join(data_path, file_name))

            if image_crop is not None:
                img = img[image_crop:-image_crop, image_crop:-image_crop]

            if img.dtype == "float32" or img.dtype == "float64":
                img = (img * 255).astype(np.uint8)

            resized_img = np.array(
                Image.fromarray(img).resize(image_size).convert("RGB")
            )

            imgs.append(resized_img)

        # yield the result
        yield np.array(imgs), y_train


def process_one_image(
    image_size: Tuple[int, int], image_crop: int, file_name: str, do_auto_trim: bool
):
    img = Image.open(file_name).convert("RGB")
    # img = mpimg.imread(file_name)
    if image_crop is not None and "hunt_plants" not in file_name:
        img = img.crop(
            box=[
                image_crop,
                image_crop,
                img.size[0] - image_crop,
                img.size[1] - image_crop,
            ]
        )

    if do_auto_trim:
        img_trimmed = auto_trim(img)
    else:
        img_trimmed = img

    if img_trimmed is None:
        print(file_name)
        return np.asarray(img.resize(image_size))

    # Resize the image
    img_trimmed = img_trimmed.resize(image_size)

    return np.asarray(img_trimmed)


def image_all_data_generator(
    data_paths: List[str],
    image_size: Tuple[int, int],
    batch_size: int = 256,
    image_crop: Optional[int] = None,
    do_auto_trim: bool = True,
    n_jobs: int = 12,
) -> Tuple[np.array, np.array]:
    """Learning generator for gans based on file paths"""
    file_names = []
    for data_path in data_paths:
        files = os.listdir(data_path)
        file_paths = [
            os.path.join(data_path, file_name)
            for file_name in files
            if ".DS_Store" not in file_name
        ]
        file_names = file_names + file_paths

    y_train = np.ones([len(file_names), 1])

    # Load in the images
    # imgs = []
    # for file_name in tqdm(file_names):
    #     img = mpimg.imread(os.path.join(data_path, file_name))
    #     if image_crop is not None:
    #         img = img[image_crop:-image_crop, image_crop:-image_crop]
    #
    #     # Resize the images
    #     resized_image = misc.imresize(img, image_size)
    #     imgs.append(resized_image)

    with multiprocessing.Pool(processes=n_jobs) as pool:
        func = partial(process_one_image, image_size, image_crop, do_auto_trim)
        imgs = pool.map(func, file_names)

    # yield the result
    yield np.array(imgs), y_train


class DCGAN:
    def __init__(
        self,
        img_rows: int,
        img_cols: int,
        channels: int,
        model_name: str,
        latent_dim: int,
        model_save_dir: str = ".",
        load_all_data: bool = True,
        rescale: bool = True,
        dropout_rate: float = 0.0,
        dropout_at_test: bool = False,
        num_base_filters: Optional[int] = None,
        load_generator_from: Optional[str] = None,
        load_discriminator_from: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_save_dir = model_save_dir
        self.load_all_data = load_all_data
        self.rescale = rescale
        self.dropout_rate = dropout_rate
        self.dropout_at_test = dropout_at_test
        self.num_base_filters = num_base_filters

        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        if load_discriminator_from is None:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(
                loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            )
        else:
            self.discriminator = load_model(load_discriminator_from)

        # Build the generator
        if load_generator_from is None:
            self.generator = self.build_generator()
        else:
            self.generator = load_model(load_generator_from)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

        # Add all necessary directories
        if self.model_save_dir != ".":
            self.check_and_add_model_dir(self.model_save_dir)
            self.check_and_add_model_dir(
                os.path.join(self.model_save_dir, self.model_name)
            )
        else:
            self.check_and_add_model_dir(self.model_name)

    def check_and_add_model_dir(self, path: str):
        """Checks to see if the passed path exists and if not creates it

        Args:
            path: The path we want to check
        """
        if not os.path.exists(os.path.join(path)):
            os.makedirs(os.path.join(path))

    def build_generator(self):
        """Creates the generator model

        Returns:
            The keras model to be used as the generator
        """

        # Add a number of layers based on the output size of the image
        # Each layer will double the total output size  of the image.
        # The number of layers is the power of two of the size of the input minus 3
        # which is from the minimum size being 2^3=8.
        number_of_layers = int(math.log(self.img_cols, 2) - 3)
        if self.num_base_filters is None:
            base_filters = 32 * 2 ** number_of_layers

        model = Sequential()
        model.add(
            Dense(base_filters * 8 * 8, activation="relu", input_dim=self.latent_dim)
        )

        # Add dropout if that is set.
        if self.dropout_rate > 0:
            if self.dropout_at_test:
                model.add(PermaDropout(self.dropout_rate))
            model.add(Dropout(self.dropout_rate))

        model.add(Reshape((8, 8, base_filters)))

        # TODO:: TRY always decreasing this with 512 for the 256 image. :shrug_
        for i in range(number_of_layers):
            model.add(UpSampling2D())
            model.add(
                Conv2D(int(base_filters / 2 ** (i + 1)), kernel_size=3, padding="same")
            )
            if self.dropout_rate > 0:
                model.add(SpatialDropout2D(self.dropout_rate))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        # Final Layer
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self) -> Model:
        """Creates the discriminator model

        Returns:
            A keras model holding the discriminator
        """

        model = Sequential()

        model.add(
            Conv2D(
                32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(
        self,
        data_generator,
        epochs: int,
        batch_size: int = 128,
        save_interval: int = 50,
        label_smoothing: bool = True,
    ) -> None:
        """Trains the generator and descriminator

        Args:
            data_generator: A data generator which returns a tuple where the first element is one batch of input and
                the second element is a vector of labels
            epochs: The number of batches to run through the model
            batch_size: The number of examples to train at once
            save_interval: The interval at which we want to save out models and images
        """
        if self.load_all_data:
            x_train, _ = data_generator.__iter__().__next__()
            valid = np.ones((batch_size, 1))
            if self.rescale:
                x_train = x_train / 127.5 - 1

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]
            if self.load_all_data:
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]
            else:
                imgs, valid = data_generator.__iter__().__next__()
                if self.rescale:
                    imgs = imgs / 127.5 - 1

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Adversarial ground truths
            if label_smoothing:
                valid = np.random.uniform(0.7, 1.0, (batch_size, 1))
                fake = np.random.uniform(0, 0.3, (batch_size, 1))
            else:
                # valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            valid = np.ones((batch_size, 1))
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (epoch, d_loss[0], 100 * d_loss[1], g_loss)
            )

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_models(epoch)
                self.save_imgs(epoch)

    def save_imgs(self, epoch: int) -> None:
        """Saves 5x5 grid of generated images labeled with the epoch

        Args:
            epoch: The epoch for which we want to save images
        """
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig(
            os.path.join(
                self.model_save_dir,
                self.model_name,
                self.model_name + "_" + str(epoch) + ".png",
            )
        )
        plt.close()

    def save_models(self, epoch: int) -> None:
        """Saves out a discriminator and a generator at the provided epoch

        Args:
            epoch: The epoch at which to save the model
        """
        self.discriminator.save(
            os.path.join(
                self.model_save_dir,
                self.model_name,
                "discriminator_{}".format(str(epoch)),
            )
        )
        self.generator.save(
            os.path.join(
                self.model_save_dir, self.model_name, "generator_{}".format(str(epoch))
            )
        )


def is_power_of_two(num: int) -> bool:
    """Returns true if the number is a power of two otherwise 0

    An intuitive proof of this is that a power of two will only have one
    on bit and all lower bits will be off, if we subtract one all lower
    bits will turn on and that bit will be off meaning that the
    bitwise and will be 0.
    Args:
        num: The number to check

    Returns:
        True if the number is a power of two and false otherwise.
    """
    return num & (num - 1) == 0


@main.command("train")
@click.argument("images-path")
@click.argument("model-name")
@click.option("--model-save-dir", type=str, default=".")
@click.option("--load-generator-from", type=str, default=None)
@click.option("--load-discriminator-from", type=str, default=None)
@click.option("--length", type=int, default=128)
@click.option("--num-base-filters", type=int, default=None)
@click.option("--batch-size", type=int, default=128)
@click.option("--save-interval", type=int, default=1000)
@click.option("--epochs", type=int, default=10001)
@click.option("--label-smoothing", type=bool, default=False)
@click.option("--dropout-rate", type=float, default=0.0)
@click.option("--dropout-at-test", type=bool, default=False)
def train(
    images_path: str,
    model_name: str,
    model_save_dir: str,
    load_generator_from: Optional[str],
    load_discriminator_from: Optional[str],
    length: int,
    num_base_filters: int,
    batch_size: int,
    save_interval: int,
    epochs: int,
    label_smoothing: bool,
    dropout_rate: float,
    dropout_at_test: bool,
) -> None:
    """Trains a generator and a discrimenator for a given image dataset.

    Args:
        images_path: The path to a folder containing all of the training images
        model_save_dir: The path to a folder where we want to save the model
        model_name: The name of the model we are creating
        length: The square dimension of the image in pixels
        batch_size: The size of the training batch
        save_interval:
        epochs: How many batches do we want to train over.
        label_smoothing: Whether to map the labels to a random number in [0, .3] for 0 and [.7, 1] for 1.
        dropout_rate: the rate of dropout in the generator
    """

    if not is_power_of_two(length) or length == 0:
        raise ValueError("the length must be a power of two")

    if length < 8:
        raise ValueError("The length must be greater than 32")

    if batch_size <= 16:
        print(
            "WARNING:: SMALL BATCH SIZE CAN RESULT IN POOR PERFORMANCE AND PREMATURE CONVERGENCE"
        )

    rows = length
    cols = length
    gen = image_random_data_generator(images_path, (rows, cols), batch_size=batch_size)
    dcgan = DCGAN(
        rows,
        cols,
        3,
        model_name,
        100,
        load_all_data=False,
        model_save_dir=model_save_dir,
        dropout_rate=dropout_rate,
        dropout_at_test=dropout_at_test,
        num_base_filters=num_base_filters,
        load_generator_from=load_generator_from,
        load_discriminator_from=load_discriminator_from,
    )
    dcgan.train(
        gen,
        epochs=epochs,
        batch_size=batch_size,
        save_interval=save_interval,
        label_smoothing=label_smoothing,
    )


if __name__ == "__main__":
    main()
