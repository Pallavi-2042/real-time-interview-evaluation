import speech_recognition as sr
import sounddevice as sd
import wave
import numpy as np

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
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")