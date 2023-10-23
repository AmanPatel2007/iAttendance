from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
from os import path
import numpy as np
from threading import Thread
import face_recognition
import numpy as np
import sqlite3
from sqlite3 import Error



global conn, capture, switch, neg, face, rec, out , checkin, checkout,images, class_names, encode_list, attendance_list, name
capture=0
switch =0
images = []
class_names = []
encode_list = []

cascade_filepath = path.abspath('Data/haarcascade_frontalface_alt.xml')
classifier = cv2.CascadeClassifier(cascade_filepath)



#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)


path = 'Images'
attendance_list = os.listdir(path)

# print(attendance_list)
for cl in attendance_list:
    cur_img = cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cl)[0])
    

    
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img)
    encodes_cur_frame = face_recognition.face_encodings(img, boxes)[0]
    # encode = face_recognition.face_encodings(img)[0]
    encode_list.append(encodes_cur_frame)
    
    

 

def detect_faces1(image: np.ndarray):
    
    #image = QtGui.QImage()
    
    _min_size = (30, 30)
    
    # haarclassifiers work better in black and white
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    faces = classifier.detectMultiScale(gray_image,
                                             scaleFactor=1.3,
                                             minNeighbors=4,
                                             flags=cv2.CASCADE_SCALE_IMAGE,
                                             minSize=_min_size)

    return faces

           
            
                
def gen_video():
    global out, capture, rec_frame,class_names, encode_list, attendance_list, name
    x=0
    y=0
    
    while True:
        success, frame = camera.read() 
        if success:
            
            
            
            #frame= detect_face(frame)
            
            faces = detect_faces1(frame)        
            
            
            for (x, y, w, h) in faces:
                frame= cv2.rectangle(frame,(x, y),(x+w, y+h),(0, 0, 255),2)
            
            frame=cv2.flip(frame,1)
            
           
                   
            
            #match faces
            faces_cur_frame = face_recognition.face_locations(frame)
            encodes_cur_frame = face_recognition.face_encodings(frame, faces_cur_frame)
            
            for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
                # Match with the known faces
                match = face_recognition.compare_faces(encode_list, encodeFace)
                face_dis = face_recognition.face_distance(encode_list, encodeFace)
                name = "unknown"
                best_match_index = np.argmin(face_dis)            
                if match[best_match_index]:
                    name = class_names[best_match_index]                    
                    #print(self.name)
                    y1, x2, y2, x1 = faceLoc
                    #below line displays name in blue color
                    frame = cv2.putText(frame, name, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    
                    
            
            
            ret, buffer = cv2.imencode('.jpg', cv2.resize(frame,(640, 480)))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            
            
       
        else:
             pass        
         
            
def CheckInEntry(checkindata):          
    
    global conn
    
    conn = sqlite3.connect("Database/DBEntry.db")
    
    
    sql = ''' INSERT INTO tblTimesheet(Name,CheckInTime,CheckOutTime, TotalTime)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, checkindata)
    conn.commit()
    return cur.lastrowid



def CheckOutEntry(checkouttime):          
    
    global conn
    
    totalTime = ""
    
    conn = sqlite3.connect("Database/DBEntry.db")
    
    #UPDATE tblTimesheet SET checkouttime = ? WHERE Name = name;
    
    sql1 = '''SELECT CheckInTime FROM tblTimesheet where name = '%s' ORDER BY CheckInTime DESC LIMIT 1''' % (name)
    cur = conn.cursor()
    cur.execute(sql1)
    conn.commit()

    records = cur.fetchall()
    checkintime = records[0][0]
    
    totalTime = datetime.datetime.strptime(checkouttime, "%m/%d/%Y %H:%M %p") - datetime.datetime.strptime(checkintime, "%m/%d/%Y %H:%M %p") 
       
    sql2 = "Update tblTimesheet set checkouttime = '%s', TotalTime = '%s' where name = '%s' and CheckInTime = '%s'" % (checkouttime, totalTime, name, checkintime)    
    cur = conn.cursor()
    cur.execute(sql2)
    conn.commit()
    
    return checkintime, totalTime               

   



@app.route('/')
def index():
    return render_template('index1.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera, conn, name
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
       
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        
        elif request.form.get('checkin') == "Check In":
            global checkin         
            
            date_time_string = datetime.datetime.now().strftime("%m/%d/%Y %H:%M %p")
            checkindata = (name,date_time_string, "", 0)
            CheckInEntry(checkindata)
                
            return render_template("index1.html", INPUT_NAME_1 = name, CHKIN_TIME = date_time_string)  
            
        elif request.form.get('checkout') == "Check Out" :
            global checkout   
            
            date_time_string = datetime.datetime.now().strftime("%m/%d/%Y %H:%M %p")
            checkoutdate = date_time_string
            checkintime, total_time_string = CheckOutEntry(checkoutdate)
      
            return render_template("index1.html", INPUT_NAME_1 = name, CHKIN_TIME = checkintime, CHKOUT_TIME = date_time_string, TOT_TIME = total_time_string)
                 
    elif request.method=='GET':
        return render_template('index1.html')
    return render_template('index1.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     