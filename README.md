# skinkr-cc
# Cloud Computing

## SkinKR API
### Overview
- API URL	https://skinkrcc-mzlcbhc7tq-uc.a.run.app
- API Version	1.0
- Request Method	Request (POST)
- Response Format	JSON Format

### Post 
- If you want to use this API, you need to post the json format containing data which is decoded from image to base64
```
requests.post("https://skinkrcc-mzlcbhc7tq-uc.a.run.app",json={'base64': *decode image to base64*})
```

### Response
- Response example
```
{
    "jenis": str(prediction),
    "deskripsi": "Blackhead adalah jenis komedo yang tampak seperti pori-pori yang membesar dan menghitam. Terjadi ketika pori-pori membesar dan terbuka ke permukaan kulit dan ke kelenjar minyak, lalu teroksidasi oleh udara dan berubah menjadi warna hitam/coklat. Komedo jenis ini adalah komedo yang paling sering dialami oleh kebanyakan orang.",
    "solusi": [
        "Menggunakan produk dengan asam salisilat",
        "Gunakan Krim dan Losion Retinoid",
        "Lakukan Chemical Peeling"
    ],
    "saran": [
        "Wardah Acnederm Pore Blackhead Balm",
        "Neutrogena Deep Clean Blackhead Eliminating Scrub",
        "Innisfree Super Volcanic Pore Clay Mask"
    ]
}
```

## Google Cloud Run
### 1. Code App (Flask, TensorFlow)
- `main.py` as implementation of app

### 2. Setup Google Cloud 
- Create new project
- Enable Cloud Run API and Cloud Build API

### 3. Install and init Google Cloud SDK
- https://cloud.google.com/sdk/docs/install

### 4. Dockerfile, .dockerignore, requirements.txt
- https://cloud.google.com/run/docs/quickstarts/build-and-deploy

### 5. Cloud build and deploy
```
gcloud builds submit --tag gcr.io/<project_id>/<function_name>
gcloud run deploy --image gcr.io/<project_id>/<function_name> --platform managed
```

### 6. Test the code
- `test/test.py` to test the code
