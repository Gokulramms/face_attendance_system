import os
import cv2
import joblib
import shutil
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file,abort,Response
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier

# Initialize Flask App
app = Flask(__name__)

# Global Constants
FACE_CASCADE_PATH = 'static/haarcascade_frontalface_default.xml'
MODEL_PATH = 'static/face_recognition_model.pkl'
FACES_DIR = 'static/faces'
ATTENDANCE_DIR = 'Attendance'

# Ensure necessary directories exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Load Haarcascade model
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Haarcascade file not found: {FACE_CASCADE_PATH}")
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Ensure today's attendance file exists
attendance_file = f'{ATTENDANCE_DIR}/Attendance-{date.today().strftime("%m_%d_%y")}.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# Helper Functions
def get_today_date():
    return date.today().strftime("%d-%B-%Y")

def get_total_registered_users():
    return len(os.listdir(FACES_DIR))

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def train_model():
    faces, labels = [], []
    for user in os.listdir(FACES_DIR):
        for img_name in os.listdir(f'{FACES_DIR}/{user}'):
            img = cv2.imread(f'{FACES_DIR}/{user}/{img_name}')
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50)).ravel()
            faces.append(resized_face)
            labels.append(user)
    if faces:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(np.array(faces), labels)
        joblib.dump(model, MODEL_PATH)

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)
    if int(userid) not in df['Roll'].values:
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

def extract_attendance():
    df = pd.read_csv(attendance_file)
    required_columns = ['Name', 'Roll', 'Time']
    df.columns = required_columns
    return df['Name'], df['Roll'], df['Time'], len(df)

def get_camera_index():
    for i in range(2):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1

# Flask Routes
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, 
                           totalreg=get_total_registered_users(), datetoday2=get_today_date())

ATTENDANCE_DIR = os.path.join(os.getcwd(), "Attendance")

def fetch_all_attendance():
    """Fetch all attendance records from multiple CSV files and combine them into one DataFrame."""
    if not os.path.exists(ATTENDANCE_DIR):
        return pd.DataFrame(columns=["#", "Name", "ID", "Time", "Date"])  # Empty table if no records exist

    all_data = []
    
    # Iterate over all CSV files in the Attendance directory
    for file in os.listdir(ATTENDANCE_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(ATTENDANCE_DIR, file)
            df = pd.read_csv(file_path)
            df["Date"] = file.replace("Attendance-", "").replace(".csv", "").replace("_", "/")  # Extract date
            all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=["#", "Name", "ID", "Time", "Date"])  # Return empty DataFrame if no data

@app.route('/download')
def download():
    """Download all attendance records as a CSV file."""
    combined_file = os.path.join(ATTENDANCE_DIR, "All_Attendance_Records.csv")

    # Fetch all attendance data and save as CSV
    df = fetch_all_attendance()
    if df.empty:
        return abort(404, description="No attendance records available.")

    df.to_csv(combined_file, index=False)

    return send_file(combined_file, as_attachment=True, mimetype='text/csv')

def identify_face(face):
    if not os.path.exists(MODEL_PATH):
        return None  # No trained model available
    
    model = joblib.load(MODEL_PATH)  # Load trained model
    prediction = model.predict(face)  # Predict the user
    return prediction

@app.route('/start', methods=['GET'])
def start():
    cam_index = get_camera_index()
    if cam_index == -1:
        return render_template('home.html', mess='No available camera found.')

    cap = cv2.VideoCapture(cam_index)
    
    detected_user = None  # Variable to store detected user

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = extract_faces(frame)
        if faces is not None:
            for (x, y, w, h) in faces:
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50)).reshape(1, -1)
                identity = identify_face(face)

                if identity is not None:
                    detected_user = identity[0]  # Store the first detected user
                    add_attendance(detected_user)  # Add attendance
                    cv2.putText(frame, detected_user, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cap.release()  # Close camera
                    cv2.destroyAllWindows()  # Close OpenCV window
                    names, rolls, times, l = extract_attendance()
                    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                                           totalreg=get_total_registered_users(), datetoday2=get_today_date(),
                                           mess=f"Attendance recorded for {detected_user}")

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:  # Exit on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('home.html', mess="No face detected. Try again.")


@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    user_folder = f'{FACES_DIR}/{newusername}_{newuserid}'
    os.makedirs(user_folder, exist_ok=True)
    cam_index = get_camera_index()
    cap = cv2.VideoCapture(cam_index)
    count = 0
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img_path = f'{user_folder}/{newusername}_{count}.jpg'
            cv2.imwrite(img_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=get_total_registered_users(), datetoday2=get_today_date())

if __name__ == '__main__':
    app.run(debug=True)