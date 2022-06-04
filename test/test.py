import requests
import base64

f = open('test/f22.jpg', 'rb')
img = f.read()
b64 = base64.b64encode(img)
print(b64)

resp = requests.post("http://127.0.0.1:5000")

# print(resp.json())