import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, request, jsonify
import base64
from io import BytesIO


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.get_json('file')
        imagefix = Image.open(BytesIO(base64.b64decode(file['base64'])))
        if file['base64'] == "":
            return jsonify({"error": "no file"})
        try: 
                model = keras.models.load_model("Model.h5")
                train_dir = os.path.join('dataset/train')
                from keras.preprocessing.image import ImageDataGenerator
                train_datagen = ImageDataGenerator(
                    rescale = 1.0/255.,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    rotation_range=40,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
                train_generator = train_datagen.flow_from_directory(train_dir,
                                                                        batch_size=27,
                                                                        class_mode='categorical',
                                                                        target_size=(224, 224))     

                class_dict = {v : k for k, v in train_generator.class_indices.items()}
                imagefix = tf.image.resize(imagefix, [224, 224])
                x = keras.preprocessing.image.img_to_array(imagefix)/255
                x = np.expand_dims(x, axis=0)
                predict = model.predict(x)
                class_prediction = np.argmax(predict)
                prediction = class_dict[class_prediction]                 
                data = {"prediction": str(prediction)}
                return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"

if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))