import requests

url = "https://model-disease.onrender.com/predict"
image_path = "download.jpeg"

with requests.Session() as session:
    with open(image_path, "rb") as img_file:
        response = session.post(url, files={"image": img_file})
    print(response.json())
