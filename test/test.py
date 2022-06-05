import requests
import base64

import base64

with open("test/p8.jpg", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read())

resp = requests.post("http://127.0.0.1:5000", json={'base64': b64_string.decode('utf-8')})
print(resp.json())
