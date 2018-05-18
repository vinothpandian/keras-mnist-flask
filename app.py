from flask import Flask, request
from flask_cors import CORS, cross_origin
from keras.preprocessing import image
from keras.models import load_model
from PIL import ImageOps
import numpy as np
from binascii import a2b_base64

app = Flask(__name__)
CORS(app, resources={
     r"/predict": {"origins": ["vinothpandian.me", "vinothpandian.github.io"]}})
model = load_model('models/mnist_model.h5')
model._make_predict_function()  # REMEMBER


def save_image(image_data, fileName):
    binary_data = a2b_base64(image_data)
    with open("fileName", "wb") as fh:
        fh.write(binary_data)


@app.route("/")
@cross_origin()
def index():
    return "Flask API for serving MNIST CNN model created using Keras"


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        fileName = "output.jpg"
        data = request.get_json()
        img_data = data.get('imageURL')
        save_image(img_data, fileName)

        img = image.load_img(fileName, grayscale=True, target_size=(28, 28))
        img = ImageOps.invert(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = np.true_divide(x, 255.)

        prediction = model.predict(x)

        return np.array2string(prediction.argmax(axis=-1))


if __name__ == "__main__":
    app.run(debug=True, )
