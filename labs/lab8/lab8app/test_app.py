import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "feature1": 5.1,
    "feature2": 3.5,
    "feature3": 1.4
    # Add more fields if needed
}

response = requests.post(url, json=payload)
print(response.json())
