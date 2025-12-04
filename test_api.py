import requests

url = "http://localhost:8000/"  # your GET endpoint

response = requests.get(url)  # use .get() since app.py is GET
print("Status code:", response.status_code)
print("Response:", response.json())
