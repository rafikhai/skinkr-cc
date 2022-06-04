from asyncio.windows_events import NULL
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
import numpy as np
from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO

def prediction(photos):
    model = keras.models.load_model("Model.h5")
    train_dir = os.path.join('dataset/train')
    validation_dir = os.path.join('dataset/test')

    train_blackhead_dir = os.path.join(train_dir, 'blackhead')
    train_folikulitis_dir = os.path.join(train_dir, 'folikulitis')
    train_melasma_dir = os.path.join(train_dir, 'melasma')
    train_nodules_dir = os.path.join(train_dir, 'nodules')
    train_papula_dir = os.path.join(train_dir, 'papula')
    train_pustula_dir = os.path.join(train_dir, 'pustula')
    train_rosacea_dir = os.path.join(train_dir, 'rosacea')
    train_whitehead_dir = os.path.join(train_dir, 'whitehead')
    train_normalface_dir = os.path.join(train_dir, 'normalface')

    # Directory with our validation acne pictures
    validation_blackhead_dir = os.path.join(validation_dir, 'blackhead')
    validation_folikulitis_dir = os.path.join(validation_dir, 'folikulitis')
    validation_melasma_dir = os.path.join(validation_dir, 'melasma')
    validation_nodules_dir = os.path.join(validation_dir, 'nodules')
    validation_papula_dir = os.path.join(validation_dir, 'papula')
    validation_pustula_dir = os.path.join(validation_dir, 'pustula')
    validation_rosacea_dir = os.path.join(validation_dir, 'rosacea')
    validation_whitehead_dir = os.path.join(validation_dir, 'whitehead')
    validation_normalface_dir = os.path.join(validation_dir, 'normalface')
    train_blackhead_fnames = os.listdir(train_blackhead_dir)
    train_folikulitis_fnames = os.listdir(train_folikulitis_dir)
    train_melasma_fnames = os.listdir(train_melasma_dir)
    train_nodules_fnames = os.listdir(train_nodules_dir)
    train_papula_fnames = os.listdir(train_papula_dir)
    train_pustula_fnames = os.listdir(train_pustula_dir)
    train_rosacea_fnames = os.listdir(train_rosacea_dir)
    train_whitehead_fnames = os.listdir(train_whitehead_dir)
    train_normalface_fnames = os.listdir(train_normalface_dir)

    model = keras.models.load_model("Model.h5")
    model.summary()
    from keras.preprocessing.image import ImageDataGenerator

    # All images will be rescaled by 1./255.
    train_datagen = ImageDataGenerator(
        rescale = 1.0/255.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen  = ImageDataGenerator(
        rescale = 1.0/255.)

    # --------------------
    # Flow training images in batches of 27 using train_datagen generator
    # --------------------
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=27,
                                                        class_mode='categorical',
                                                        target_size=(224, 224))     
    # --------------------
    # Flow validation images in batches of 22 using test_datagen generator
    # --------------------
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=22,
                                                            class_mode  = 'categorical',
                                                            target_size = (224, 224))

    class_dict = {v : k for k, v in train_generator.class_indices.items()}
    img = keras.preprocessing.image.load_img(
        photos,
        target_size=(224, 224)
    )
    x = keras.preprocessing.image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x)
    class_prediction = np.argmax(predict)
    prediction = class_dict[class_prediction]

    print('percentage:')
    print(prediction)
    return prediction
    
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.get_json()
        if file is NULL:
            return jsonify({"error": "no file"})
        try:            
            image_bytes = Image.open(BytesIO(base64.b64decode(file['base64'])))
            tensor = prediction(image_bytes)
            data = {"prediction": str(tensor)}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
