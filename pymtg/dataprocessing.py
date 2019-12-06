import json
import os
import requests
from typing import Dict, List, Optional
import urllib

from bs4 import BeautifulSoup
import pandas as pd


def parse_mtg_json(
    path: str,
    start_token: str = "<|startoftext|>",
    end_token: str = "<|endoftext|>",
    filter_card_type: Optional[str] = None,
    filter_mana: Optional[List[str]] = None,
) -> str:
    """Parses out the salient fields from magic cards json found here:

    Args:
        path: The path to the json dump
        start_token: The token to delimit every card with
        end_token: The token to delimit the end of every card with.
        filter_card_type: Only selects data of that card type
        filter_mana: Only selects data of that mana type

    Returns:
        A string holding the compiled data
    """
    with open(path, encoding="ISO-8859-1") as json_file:
        j_string = json_file.read()
        parsed = json.loads(j_string)

    keys = list(parsed.keys())
    training_text = []
    for i, key in enumerate(keys):
        current_keys = parsed[key]
        name = "name|| " + parsed[key]["name"] if "name" in current_keys else ""
        colors = (
            "colors||" + "|".join(parsed[key]["colors"])
            if "colors" in current_keys
            else ""
        )
        mana_cost = (
            "mana||" + parsed[key]["manaCost"] if "manaCost" in current_keys else ""
        )
        subtypes = (
            "subtypes||" + "|".join(parsed[key]["subtypes"])
            if "subtypes" in current_keys
            else ""
        )
        card_type = "type||" + parsed[key]["type"] if "type" in current_keys else ""
        types = (
            "types||" + "|".join(parsed[key]["types"])
            if "types" in current_keys
            else ""
        )
        text = "text||" + parsed[key]["text"] if "text" in current_keys else ""

        toughness = (
            "toughness||" + parsed[key]["toughness"]
            if "toughness" in current_keys
            else ""
        )
        power = "power||" + parsed[key]["power"] if "power" in current_keys else ""

        # Filter out cards which are not the correct type
        if filter_card_type is not None and filter_card_type not in card_type:
            continue

        row = "\n".join(
            [
                start_token,
                name,
                colors,
                mana_cost,
                card_type,
                types,
                subtypes,
                toughness,
                power,
                text,
                end_token,
            ]
        )
        training_text.append(row)

    return "\n".join(training_text)


def create_path(file_path: str) -> None:
    # Create the path if it doesn't exist
    path = os.path.split(file_path)[0]
    if not os.path.exists(path):
        os.makedirs(path)


def download_text_data(file_path: str) -> None:
    """This method downloads all of the text data used for training the model into the data/text folder
    """
    if not os.path.isfile(file_path):
        # Create the path if it doesn't exist
        create_path(file_path)

        # Download the data
        url = "https://www.mtgjson.com/files/AllCards.json"
        data = requests.get(url)
        with open(file_path, "w") as f:
            f.write(data.text)


def get_html(url: str, headers: Optional[Dict[str, str]] = None):
    """Gets the beautiful soup parse of the provided website

    Args:
        url: The url we want to parse
        headers: The request headers

    Returns:
        Beautiful soup parse of the website
    """

    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
        }
    req = urllib.request.Request(url=url, headers=headers)
    with urllib.request.urlopen(req) as response:
        html = response.read()
        # Create Beautiful Soup Object Model
        soup = BeautifulSoup(html, "html.parser")
    return soup


def scrape_image_data(file_path: str) -> None:
    """Saves the urls of all images used for training mtg gans these can then be downloaded by running
    wget -i magic_urls.csv
    """
    base_url = "https://www.mtgpics.com/"
    search_extension = "art?pointeur="
    index = 0

    # Define headers for accessing websites
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
    }
    urls = []

    for index in range(0, 427 * 60, 60):
        soup = get_html(base_url + search_extension + str(index), headers)
        for div in soup.find_all("div"):
            try:
                if "url(" in div["style"]:
                    urls.append(base_url + div["style"].split("url(")[1][:-2])
            except:
                continue

    pd.DataFrame(urls, columns=["url"]).to_csv(file_path, index=False, header=False)
