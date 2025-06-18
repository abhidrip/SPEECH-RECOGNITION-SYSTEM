import tkinter as tk
from tkinter import ttk
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# === CONFIG ===
AUDIO_FILE = "live_input.wav"
fs = 16000
recording = False
recorded_frames = []

# === Load faster-whisper model ===
print("Loading model...")
model = WhisperModel("base")
print("✅ Model loaded!")

# === Record audio ===
def audio_callback(indata, frames, time, status):
    if recording:
        recorded_frames.append(indata.copy())

def handle_record():
    global recording, recorded_frames
    recorded_frames = []
    recording = True
    status_label.config(text="🎙️ Recording... Speak now.")
    threading.Thread(target=start_stream).start()

def start_stream():
    with sd.InputStream(callback=audio_callback, samplerate=fs, channels=1):
        while recording:
            sd.sleep(100)

def handle_transcribe():
    global recording
    if not recording:
        status_label.config(text="❗ Not recording.")
        return
    recording = False
    status_label.config(text="🧠 Transcribing...")
    threading.Thread(target=process_transcription).start()

def process_transcription():
    try:
        audio_np = np.concatenate(recorded_frames, axis=0)
        sf.write(AUDIO_FILE, audio_np, fs)

        segments, info = model.transcribe(AUDIO_FILE)
        transcription = " ".join([segment.text for segment in segments])

        result_box.delete(1.0, tk.END)
        result_box.insert(tk.END, transcription)
        status_label.config(text="✅ Transcription complete.")
    except Exception as e:
        status_label.config(text=f"❌ Error: {e}")

# === Build UI ===
root = tk.Tk()
root.title("🎙️ speech-to-text 🎉")
root.geometry("700x500")
root.configure(bg="#f5f5dc")  # Light beige background

# === FIRST: place the canvas ===
canvas = tk.Canvas(root, width=700, height=500, bg="#f5f5dc", highlightthickness=0)
canvas.place(x=0, y=0)
canvas.create_oval(50, 50, 150, 150, fill="#ffe4e1", outline="")
canvas.create_oval(550, 50, 650, 150, fill="#ffe4e1", outline="")

# === THEN: pack other widgets ===
style = ttk.Style()
style.theme_use('clam')

primary_color = "#ff69b4"
secondary_color = "#20b2aa"
text_color = "#333333"
font_main = ("Comic Sans MS", 16, "bold")

style.configure('TButton',
                font=font_main,
                foreground="white",
                background=primary_color,
                padding=15,
                relief="flat")
style.map('TButton',
          background=[('active', secondary_color)])

status_label = ttk.Label(root, text="Press 'Start Recording' to begin",
                         font=font_main,
                         foreground=text_color,
                         background="#f5f5dc")
status_label.pack(pady=20)

record_btn = ttk.Button(root, text="🎤 Start Recording", command=handle_record)
record_btn.pack(pady=20)

transcribe_btn = ttk.Button(root, text="🧠 Start Transcribing", command=handle_transcribe)
transcribe_btn.pack(pady=20)

result_box = tk.Text(root, height=10, width=70, font=("Comic Sans MS", 14), bg="#fff0f5")
result_box.pack(pady=20)

root.mainloop()
