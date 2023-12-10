from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from keras.preprocessing import image
import imutils


app = Flask(__name__)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

model = load_model('models\model_brain_tumor.h5')

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/checkup', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message="No selected file")

        if file:
            pil_image = Image.open(file)
            external_image_cropped = crop_imgs(set_name=np.array([pil_image]))[0]
            external_image_resized = preprocess_imgs(set_name=np.array([external_image_cropped]), img_size=(224, 224))[0]
            external_image_prediction = model.predict(np.array([external_image_resized]))
            predicted_class = "YES" if external_image_prediction > 0.5 else "NO"
            return render_template('result.html', predicted_class=predicted_class)

    return render_template('index.html', message="Please upload a MRI Image of Brain")

if __name__ == '__main__':
    app.run(debug=True)
