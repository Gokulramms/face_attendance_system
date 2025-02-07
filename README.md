# Face Recognition Attendance System

## Introduction
The **Face Recognition Attendance System** is a Flask-based application that utilizes OpenCV and machine learning to recognize faces and mark attendance automatically. It captures faces, stores them, trains a model, and records attendance in CSV format.

## Features
- **Face Detection & Recognition**: Uses OpenCV's Haarcascade to detect and identify faces.
- **Automated Attendance Recording**: Marks attendance with timestamp when a registered face is detected.
- **Export Attendance Data**: Download daily attendance reports as CSV files.
- **User Registration**: Allows new users to register by capturing face images and training the model.
- **Live Camera Feed**: Uses webcam to capture real-time face recognition.
- **Attendance History**: Stores past attendance records for review.

## Technologies Used
- **Python** (Flask, OpenCV, NumPy, Pandas, scikit-learn)
- **Flask** (Backend framework for web application)
- **OpenCV** (Face detection & recognition)
- **scikit-learn** (K-Nearest Neighbors model for face recognition)
- **Pandas** (For handling attendance records)
- **HTML/CSS** (Frontend UI)

## Installation

### Prerequisites
Ensure you have **Python 3.8+** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Gokulramms/face-recognition-attendance.git
cd face-recognition-attendance
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Open in Browser
Visit `http://127.0.0.1:5000/` to access the application.

## Usage

### 1. Register a New User
- Go to the **Add User** section.
- Enter Name and Roll Number.
- The system will capture face images and train the model.

### 2. Start Attendance
- Click on **Start Attendance** to begin face recognition.
- The system will detect faces and mark attendance automatically.

### 3. Download Attendance
- Click **Export Attendance** to download the CSV file.

### 4. View Attendance History
- Navigate to **Attendance History** to see past records.

## Project Structure
```
face-recognition-attendance/
│── app.py                  # Main Flask Application
│── templates/
│   ├── home.html           # Frontend UI
│── static/
│   ├── faces/              # Stored Face Images
│   ├── haarcascade_frontalface_default.xml  # Face Detection Model
│   ├── face_recognition_model.pkl           # Trained ML Model
│── Attendance/             # Attendance CSV Records
│── requirements.txt        # Required Python Libraries
│── README.md               # Project Documentation
```

## Known Issues & Fixes
- **Camera Not Detected**: Ensure your webcam is connected and functional.
- **Model Not Recognizing Faces**: Ensure the training process captures enough images.
- **Attendance File Not Found**: The system will create a new file if none exists.

## Future Enhancements
- Add a **database** for better data storage.
- Implement **mobile app** integration.
- Enhance **UI design** with modern frameworks.

## Contributors
- **Gokulramm S** (Developer)

## License
This project is licensed under the MIT License.

