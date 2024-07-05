from flask import Flask, render_template, request, Response, send_file, redirect, url_for, jsonify, request, render_template
from werkzeug.utils import secure_filename
import cv2
import os
import base64
import numpy as np
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/upload_image/"
current_image = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
path ="F:/2024/ML_Project_Repo/03 Python Sketches/haarcascade_mcs_nose.xml"
nose_cascade = cv2.CascadeClassifier(path)
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
        return jsonify({'message': 'Uploaded Input image', 'image': jpg_as_text}), 200

@app.route('/gray_scale',methods=['POST'])
def gray_scale():
    global current_image
    img = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    text_data = send_as_text(img)
    return jsonify({'message': 'gray_scale is applied', 'image': text_data}), 200



@app.route('/canny',methods=['POST'])
def canny():
    global current_image
    t_lower = 50
    t_higher =150
    edge = cv2.Canny(current_image,t_lower,t_higher)
    text_data = send_as_text(edge)
    return jsonify({'message': 'canny is applied', 'image': text_data}), 200

@app.route('/gaussian',methods=['POST'])
def gaussian():
    global current_image
    output = cv2.GaussianBlur(current_image,(7,7),0)
    text_data = send_as_text(output)
    return jsonify({'message': 'gaussian is applied', 'image': text_data}), 200

@app.route('/median',methods=['POST'])
def median():
    global current_image
    output = cv2.medianBlur(current_image,5)
    text_data = send_as_text(output)
    return jsonify({'message': 'median is applied', 'image': text_data}), 200


@app.route('/bilateral',methods=['POST'])
def bilateral():
    global current_image
    output = cv2.bilateralFilter(current_image,9,75,75)
    text_data = send_as_text(output)
    return jsonify({'message': 'bilateral is applied', 'image': text_data}), 200

@app.route('/sharpening',methods=['POST'])
def sharpening():
    global current_image
    kernel = np.array([[-1,-1,-1],
                        [-1,9,-1],
                        [-1,-1,-1]])
    output = cv2.filter2D(current_image,-1, kernel)
    text_data = send_as_text(output)
    return jsonify({'message': 'sharpening is applied', 'image': text_data}), 200


@app.route('/sobel_edge',methods=['POST'])
def sobel_edge():
    global current_image
    sobel_x = cv2.Sobel(current_image,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(current_image,cv2.CV_64F,0,1,ksize=5)
    
    output = cv2.magnitude(sobel_x,sobel_y)
    text_data = send_as_text(output)
    return jsonify({'message': 'Sobel Effect is applied', 'image': text_data}), 200

@app.route('/face_detect',methods=['POST'])
def face_detect():
    global current_image
    
    faces = detect_faces(current_image)
 
    for (x,y,w,h) in faces:
        
        cv2.rectangle(current_image,(x,y),(x+w,y+h),(0,0,255),2)
    
    text_data = send_as_text(current_image)
    return jsonify({'message': f'Total faces: {len(faces)}', 'image': text_data}), 200

@app.route('/eye_detect',methods=['POST'])
def eye_detect():
    global current_image
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(current_image)

    for (x,y,w,h) in faces:
        
       
        
        roi_rgb = current_image[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor =1.1, minNeighbors = 10, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(current_image,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
    text_data = send_as_text(current_image)
    return jsonify({'message': f'Eyes detected', 'image': text_data}), 200


@app.route('/nose_detect',methods=['POST'])
def nose_detect():
    global current_image
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(current_image)
 
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
      
        nose = nose_cascade.detectMultiScale(gray, scaleFactor =1.1,  minNeighbors=5, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(current_image,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
    text_data = send_as_text(current_image)
    return jsonify({'message': f'Nose detected', 'image': text_data}), 200


@app.route('/mouth_detect',methods=['POST'])
def mouth_detect():
    global current_image
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(current_image)
 
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        mouth = mouth_cascade.detectMultiScale(gray, scaleFactor =1.1,  minNeighbors=110, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
        for (nx, ny, nw, nh) in mouth:
            cv2.rectangle(current_image,(nx,ny),(nx+nw,ny+nh),(255,0,255),2)
    text_data = send_as_text(current_image)
    return jsonify({'message': f'mouth detected', 'image': text_data}), 200




@app.route('/upperbody_detect',methods=['POST'])
def upperbody_detect():
    global current_image
    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(current_image)
 
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        cv2.rectangle(current_image,(x,y),(x+w,y+h),(0,255,0),8)
        upper_body = upper_body_cascade.detectMultiScale(gray, scaleFactor =1.1,  minNeighbors=110, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        for (nx, ny, nw, nh) in upper_body:
            cv2.rectangle(current_image,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)
    text_data = send_as_text(current_image)
    return jsonify({'message': f'Upper Body detected', 'image': text_data}), 200



def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor =1.1, minNeighbors = 8, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces
def send_as_text(img):
    _,buffer = cv2.imencode('.jpg',img) 
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)