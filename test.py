import requests

url = 'http://localhost:5000/predict'

files = {'image': open('Control.png', 'rb')}
print(files)

response = requests.post(url, files=files)
print(response.json())