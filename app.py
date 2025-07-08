from flask import Flask, render_template, url_for, request, Response, jsonify, redirect
import sqlite3
import cv2
import os
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import speech_recognition as sr
import sounddevice as sd
import wave
import numpy as np

def Recording():
    if os.path.exists('recording.txt'):
        os.remove('recording.txt')

    while True:
        duration = 5
        fs = 44100
        channels=2
        filename="input_audio.wav"

        print("Recording...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype=np.int16)
        sd.wait()
        print("Recording complete.")

        # Save the recorded audio to a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())

        # Perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language='en')
                print("Recognized text:", text)
                f = open('recording.txt', 'w')
                f.write(text)
                f.close()
            except sr.UnknownValueError:
                print("Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

command = """CREATE TABLE IF NOT EXISTS scores(name TEXT, score TEXT, status TEXT, graph TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/adminlog', methods=['GET', 'POST'])
def adminlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        if name == 'pallavi' and password == 'pallavi123':
            cursor.execute("SELECT * FROM scores")
            result = cursor.fetchall()
            return render_template('adminlog.html', result=result)
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            os.system('python recognition.py')
            f = open('recognise.txt', 'r')
            name2 = f.read()
            f.close()
            if name == name2:
                t1 = threading.Thread(target=Recording)
                t1.start()
                return redirect(url_for('home'))
            else:
                return render_template('index.html', msg='Sorry, face mismatched,  Try Again')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        f = open('session.txt', 'w')
        f.write(name)
        f.close()
        os.system('python dataset.py')
        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return redirect(url_for('userlog'))
    
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('index.html', msg = "Thank You")

@app.route('/HeadPoseStream')
def HeadPoseStream():

    from ai_proctoring import HeadPose
    return Response(HeadPose(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/MouthPoseStream')
def MouthPoseStream():

    from ai_proctoring import MouthPose
    return Response(MouthPose(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/facePoseStream')
def facePoseStream():

    from ai_proctoring import facePose
    return Response(facePose(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/eyePoseStream')
def eyePoseStream():

    from ai_proctoring import eyePose
    return Response(eyePose(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/PersonStream')
def PersonStream():

    from ai_proctoring import Person
    return Response(Person(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/PhoneStream')
def PhoneStream():

    from ai_proctoring import Phone
    return Response(Phone(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/EmotionStream')
def EmotionStream():

    from ai_proctoring import Emotion
    return Response(Emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/exam",methods=["POST","GET"])
def exam():
    if request.method == 'POST':
        data = request.form
        print("===================================")
        print(data)
        user_answers = []
        for key in data:
            user_answers.append(int(data[key]))
            
        print(user_answers)
        answers = [3,4,1,1,3]
        score = 0
        
        for i in range(5):
            if user_answers[i] == answers[i]:
                score = score + 20
        print(score)

        if score >= 80:
            text="You Are Excellent !!"
        elif (score >= 40 and score < 80):
            text="You Can Be Better !!"
        else:
            text="You Should Work Hard !!"

        f = open('recognise.txt', 'r')
        name = f.read()
        f.close()

        try:
            # Load CSV (assumes one column with emotion labels)
            df = pd.read_csv('emotions.csv')  # replace with your actual filename

            # If the column isn't named, you may need to specify: df.iloc[:, 0]
            emotions = df.iloc[:, 0].tolist()

            # Count each emotion
            emotion_counts = Counter(emotions)

            # Plot pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', startangle=140)
            plt.title('Emotion Distribution')
            plt.axis('equal')  # Ensures pie is circular
            plt.savefig('static/'+name+'.png')
        except:
            pass
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        file = 'static/'+name+'.png'
        cursor.execute("INSERT INTO scores VALUES ('"+name+"', '"+str(score)+"', '"+text+"', '"+file+"')")
        connection.commit()
            
        print("===================================")
        f = open('stream.txt', 'w')
        f.write('stop')
        f.close()

        res = f'Thank You! your score is {score} and {text}'
        return jsonify(res)
    return jsonify("error")

@app.route("/persons")
def persons():
    if os.path.exists('personnum.txt'):
        f = open('personnum.txt', 'r')
        num = f.read()
        f.close()
        return jsonify(num)
    else:
        return jsonify('0')

@app.route("/Speech")
def Speech():
    if os.path.exists('recording.txt'):
        f = open('recording.txt', 'r')
        num = f.read()
        f.close()
        os.remove('recording.txt')
        return jsonify(num)
    else:
        return jsonify(' ')
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
