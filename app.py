from flask import Flask,render_template,request, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
DEM = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

model = load_model('./static/modelMobileNetFlower.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    global DEM
    name_bird = ''
    img = request.files['image']

    img.save('static/{}.jpg'.format(DEM))    
    arr = cv2.imread('static/{}.jpg'.format(DEM))

    arr = cv2.resize(arr, (170,170))
    arr = arr / 255.0
    arr = arr.reshape(1, 170,170,3)
    prediction = model.predict(arr)

    index = np.argmax(prediction)
    if index == 0:
        name_bird = 'Bồ câu'
    elif index == 1:
        name_bird = 'Chào mào'
    elif index == 2:
        name_bird ='Chích choè'
    elif index == 3:
        name_bird = 'Hoạ mi'
    elif index == 4:
        name_bird = 'Quạ'
    elif index == 5:
        name_bird = 'Chim sẻ'
    elif index == 6:
        name_bird = 'Vàng anh'
    else:
        name_bird = 'Vẹt'

    max = np.max(prediction)
    max = round(max,2)
    DEM += 1
    object = {'name': name_bird, 'max' : max}
    return render_template('predict.html',object = object,index=index)


@app.route('/load_img')
def load_img():
    global DEM
    return send_from_directory('static', "{}.jpg".format(DEM-1))


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080) 


