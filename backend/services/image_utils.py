import requests
from PIL import Image
from io import BytesIO


def download_image(image_url: str, save_path: str) -> None:
    """
    Download an image from a URL and save it to a local path.
    Args:
        image_url (str): The URL of the image to download.
        save_path (str): The local file path to save the image.
    """
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image.save(save_path) 