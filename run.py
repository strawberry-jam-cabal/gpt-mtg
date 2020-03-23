"""
This script has a few functionaities:

1. download-images:  This call will download all images necessary for training the gan
2. process-card-text: This call will download the json representation of all of the cards data and process it into a single
    text file.

"""
from typing import Optional

import os
import sys
import json
import urllib.request

import gpt_2_simple as gpt2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from pymtg import dataprocessing

if "linux" in sys.platform:
    os.environ["LC_AL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
else:
    os.environ["LC_AL"] = "en_US.utf-8"
    os.environ["LANG"] = "en_US.utf-8"

import click


@click.group()
def main():
    pass


@main.command("compile-sets-to-json")
@click.argument("path")
@click.argument("output-path")
def compile_sets_to_json(path: str, output_path: str):
    """Compiles all of the set json to a single json file with a list as the top level.

    Args:
        path: to the folder containing all json set files.

    Returns:
    """
    sets = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            sets.append(json.load(f))

    with open(output_path, "w") as outfile:
        json.dump(sets, outfile)


@main.command("download-images")
@click.option("--images-path", type=str, default="./data/images")
def download_images(images_path: str):
    """Downloads all of the images necessary for training an MTG GAN

    Args:
        images_path: The path to a directory which will hold all images

    Returns:

    """

    # Scrape the image data and store the urls in a text file one line for each url
    url_path = "./data/magic_urls.csv"
    if not os.path.isfile(url_path):
        dataprocessing.scrape_image_data(url_path)

    # Read in the url paths and download the images one at a time.
    # since I split the path in create_path I need a dummy file.
    dataprocessing.create_path(os.path.join(images_path, "x"))
    with open(url_path, "r") as f:
        for i, url in enumerate(f):
            urllib.request.urlretrieve(url, os.path.join(images_path, str(i) + ".png"))


@main.command("process-card-text")
@click.option("--input-file-path", type=str, default="./data/AllCards.json")
@click.option("--output-file-path", type=str, default="./data/mtg_combined.txt")
def process_card_text(input_file_path: str, output_file_path) -> None:
    """Gets all necessary text data for MGT GPT2 and compiles it into one .txt file

    Args:
        input_file_path: The path to the MTG json file with all cards
        output_file_path: The output path to where the compiled .txt file should be
    """

    # Download all the cards if we don't have them
    dataprocessing.create_path(input_file_path)
    dataprocessing.download_text_data(input_file_path)

    # Compile all of the cards to a combined text file for gpt2
    text = dataprocessing.parse_mtg_json(input_file_path)
    dataprocessing.create_path(output_file_path)
    with open(output_file_path, "w") as f:
        f.write(text)


@main.command("finetune")
@click.option(
    "--model-name", type=str, default="124M", help="Can be 117M, 124M, or 355M"
)
@click.option("--text-path", type=str, default="./data/mtg_combined.txt")
@click.option("--checkpoint-dir", type=str, default="checkpoint")
@click.option("--num-steps", type=int, default=3000)
@click.option("--sample-length", type=int, default=1023)
@click.option("--save-every", type=int, default=None)
# TODO:: aDD TEXT SIZE
def finetune(
    model_name: str,
    text_path: str,
    checkpoint_dir: str,
    num_steps: int,
    sample_length: int,
    save_every: Optional[int],
) -> None:

    # Download the model if it is not present
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

    sess = gpt2.start_tf_sess()

    if save_every is None:
        save_every = int(num_steps / 4)

    gpt2.finetune(
        sess,
        text_path,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        steps=num_steps,
        sample_length=sample_length,
        save_every=save_every,
    )  # steps is max number of training steps

    gpt2.generate(sess)


@main.command("generate-text")
@click.option("--checkpoint-dir", type=str, default="checkpoint")
@click.option("--n-samples", type=int, default=1)
@click.option("--batch-size", type=int, default=1)
@click.option("--length", type=int, default=1023)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="lower more consistent, higher more fun",
)
@click.option("--destination-path", type=str, default=None)
@click.option("--prefix", type=str, default=None)
def generate_text(
    checkpoint_dir: str,
    n_samples: int,
    batch_size: int,
    length: int,
    temperature: float,
    destination_path: str,
    prefix: Optional[str],
) -> None:
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, checkpoint_dir=checkpoint_dir)
    gpt2.generate(
        sess,
        checkpoint_dir=checkpoint_dir,
        nsamples=n_samples,
        batch_size=batch_size,
        length=length,
        temperature=temperature,
        destination_path=destination_path,
        prefix=prefix,
    )


@main.command("generate-images")
@click.argument("model-path")
@click.argument("output-dir")
@click.option("--num-images", type=int, default=50)
def generate_images(model_path: str, output_dir: str, num_images: int):
    plt.ioff()
    model = load_model(model_path)
    noise = np.random.normal(0, 1, (num_images, 100))
    imgs = model.predict(noise)
    for i in range(0, 50):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(imgs[i])
        ax.axis("off")
        fig.savefig(os.path.join(output_dir, "mtg_{}.png".format(str(i))))
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(0.5 * imgs[i] + 0.5)  # Need to normalize images to proper range.
        ax.axis("off")
        fig.savefig(os.path.join(output_dir, "mtg_{}_n.png".format(str(i))))
        plt.close(fig)


if __name__ == "__main__":
    main()
