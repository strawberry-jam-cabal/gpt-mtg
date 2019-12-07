"""
This script has a few functionaities:

1. download-images:  This call will download all images necessary for training the gan
2. process-card-text: This call will download the json representation of all of the cards data and process it into a single
    text file.
"""

import os
import sys
import urllib.request

import gpt_2_simple as gpt2

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
@click.option("--num-steps", type=int, default=10000)
def finetune(model_name: str, text_path: str, num_steps) -> None:

    # Download the model if it is not present
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

    sess = gpt2.start_tf_sess()
    gpt2.finetune(
        sess, text_path, model_name=model_name, steps=num_steps
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
    default=0.7,
    help="lower more consistent, higher more fun",
)
def generate_text(
    checkpoint_dir: str,
    n_samples: int,
    batch_size: int,
    length: int,
    temperature: float,
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
    )


if __name__ == "__main__":
    main()
