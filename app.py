import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import sqlite3
import cv2
import numpy as np
import face_recognition

def resource_path(relative_path):
    """Get the absolute path to a resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = Flask(__name__)
app.secret_key = 'your_secret_key'
DATABASE = resource_path("attendance.db")
UPLOAD_FOLDER = resource_path("static/uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Rest of your code...

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.before_first_request
def setup_database():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                reg_no TEXT NOT NULL,
                email TEXT,
                image TEXT,
                encoding BLOB
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date TEXT,
                status TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        ''')

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    students = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    return render_template('index.html', students=students)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['username'] = username
            return redirect(url_for('index'))
        flash('Invalid credentials, please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        reg_no = request.form['reg_no']
        email = request.form['email']
        file = request.files['image']
        
        if file and name and reg_no and email:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                conn = get_db_connection()
                conn.execute('INSERT INTO students (name, reg_no, email, image, encoding) VALUES (?, ?, ?, ?, ?)',
                             (name, reg_no, email, filename, np.array(encodings[0]).tobytes()))
                conn.commit()
                conn.close()
                flash('Student added successfully.', 'success')
                return redirect(url_for('index'))
            else:
                flash('No face detected in the image. Please upload a clear image.', 'warning')
        else:
            flash('All fields are required.', 'danger')
    return render_template('register.html')

@app.route('/manage_students')
def manage_students():
    conn = get_db_connection()
    students = conn.execute('SELECT * FROM students').fetchall()
    conn.close()
    return render_template('manage_students.html', students=students)

@app.route('/edit_student/<int:id>', methods=['GET', 'POST'])
def edit_student(id):
    conn = get_db_connection()
    student = conn.execute('SELECT * FROM students WHERE id = ?', (id,)).fetchone()

    if request.method == 'POST':
        name = request.form['name']
        reg_no = request.form['reg_no']
        email = request.form['email']
        conn.execute('UPDATE students SET name = ?, reg_no = ?, email = ? WHERE id = ?', 
                     (name, reg_no, email, id))
        conn.commit()
        conn.close()
        flash('Student updated successfully.', 'success')
        return redirect(url_for('manage_students'))
    
    conn.close()
    return render_template('edit_student.html', student=student)

@app.route('/attendance')
def attendance():
    conn = get_db_connection()
    records = conn.execute('''
        SELECT a.id, s.name, s.reg_no, a.date, a.status 
        FROM attendance a 
        JOIN students s ON a.student_id = s.id
    ''').fetchall()
    conn.close()
    return render_template('attendance.html', records=records)

from flask import Response
import time

def gen_frames():
    conn = get_db_connection()
    students = conn.execute('SELECT id, name, encoding FROM students').fetchall()
    known_face_encodings = [np.frombuffer(student['encoding'], dtype=np.float64) for student in students]
    known_face_ids = [student['id'] for student in students]
    known_face_names = [student['name'] for student in students]
    video_capture = cv2.VideoCapture(0)
    attendance_log = []
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    student_id = known_face_ids[best_match_index]
                    student_name = known_face_names[best_match_index]
                    date = datetime.datetime.now().strftime("%Y-%m-%d")

                    existing_record = conn.execute(
                        'SELECT * FROM attendance WHERE student_id = ? AND date = ?',
                        (student_id, date)
                    ).fetchone()

                    if not existing_record:
                        conn.execute('INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)', 
                                     (student_id, date, 'Present'))
                        conn.commit()
                        print(f"{student_name} marked present for {date}")
                    else:
                        print(f"{student_name} already marked present for {date}")

            for (top, right, bottom, left) in face_locations:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
    conn.close()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')



if __name__ == '__main__':
    app.run(debug=True)
