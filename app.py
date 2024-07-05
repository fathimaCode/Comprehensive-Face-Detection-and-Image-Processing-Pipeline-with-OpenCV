from flask import Flask, render_template, request, Response, send_file, redirect, url_for, jsonify, request, render_template
from werkzeug.utils import secure_filename
import cv2
import os
import base64
import numpy as np
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/upload_image/"
current_image = None
@app.route('/')
def root():  
    return render_template("index.html")


@app.route('/upload-image',methods=['POST'])
def upload_imag():
    global current_image
    if 'image' not in request.files:
        return jsonify({'error':'No File part'}),400
    file = request.files['image']
    if file.filename=='':
        return jsonify({'error':'No File part'}),400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(file_path)
        print(file_path)
        input_image = cv2.imread(file_path)
        print(input_image)
        current_image = input_image
        
        _,buffer = cv2.imencode('.jpg',input_image) 
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'message': 'File uploaded successfully', 'image': jpg_as_text}), 200

@app.route('/gray_scale',methods=['POST'])
def gray_scale():
    global current_image
    img = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    text_data = send_as_text(img)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200



@app.route('/canny',methods=['POST'])
def canny():
    global current_image
    t_lower = 50
    t_higher =150
    edge = cv2.Canny(current_image,t_lower,t_higher)
    text_data = send_as_text(edge)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200

@app.route('/gaussian',methods=['POST'])
def gaussian():
    global current_image
    output = cv2.GaussianBlur(current_image,(7,7),0)
    text_data = send_as_text(output)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200

@app.route('/median',methods=['POST'])
def median():
    global current_image
    output = cv2.medianBlur(current_image,5)
    text_data = send_as_text(output)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200


@app.route('/bilateral',methods=['POST'])
def bilateral():
    global current_image
    output = cv2.bilateralFilter(current_image,9,75,75)
    text_data = send_as_text(output)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200

@app.route('/sharpening',methods=['POST'])
def sharpening():
    global current_image
    kernel = np.array([[-1,-1,-1],
                        [-1,9,-1],
                        [-1,-1,-1]])
    output = cv2.filter2D(current_image,-1, kernel)
    text_data = send_as_text(output)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200


@app.route('/sobel_edge',methods=['POST'])
def sobel_edge():
    global current_image
    sobel_x = cv2.Sobel(current_image,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(current_image,cv2.CV_64F,0,1,ksize=5)
    
    output = cv2.magnitude(sobel_x,sobel_y)
    text_data = send_as_text(output)
    return jsonify({'message': 'File uploaded successfully', 'image': text_data}), 200



def send_as_text(img):
    _,buffer = cv2.imencode('.jpg',img) 
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)