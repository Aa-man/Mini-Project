import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pyttsx3
import face_recognition

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Global variables
entry_name = None
entry_age = None
gender_var = None
form_widgets = []

dataset_path = "C:/Users/AMAN SINGH/Desktop/MiniProject/dataset"
os.makedirs(dataset_path, exist_ok=True)

# ---------- FACE CAPTURE ----------
def capture_faces_gui(name, age, gender):
    person_folder = os.path.join(dataset_path, f"{name}_{age}_{gender}")
    os.makedirs(person_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    max_images = 30

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Option 1: Save the full frame
            file_path = os.path.join(person_folder, f"{name}_{count+1}.jpg")
            cv2.imwrite(file_path, frame)
            count += 1

            # Show rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Capturing {count}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    engine.say(f"Face captured for {name}")
    engine.runAndWait()
    messagebox.showinfo("Success", f"{count} face images captured for {name}!")


# ---------- FACE RECOGNITION ----------
def start_recognition():
    known_encodings = []
    known_labels = []

    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            print(f"Reading image: {path}")
            image = cv2.imread(path)
            if image is None:
                print(f"Skipped unreadable image: {path}")
                continue

            # Handle grayscale or BGRA
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

            face_locations = face_recognition.face_locations(rgb)
            if len(face_locations) == 0:
                print(f"No face found in {path}")
                continue

            encs = face_recognition.face_encodings(rgb, face_locations)
            if encs:
                known_encodings.append(encs[0])
                known_labels.append(person)
                print(f"Encoding added for {person}")
            else:
                print(f"Failed to encode {path}")

    if not known_encodings:
        messagebox.showerror("Error", "No valid faces in dataset.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera.")
        return

    threshold = 0.6  # More flexible match threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)

        for enc, (top, right, bottom, left) in zip(encs, locs):
            dists = face_recognition.face_distance(known_encodings, enc)
            print("Distances:", dists)

            if len(dists) == 0:
                name = "Unknown"
            else:
                match = np.argmin(dists)
                name = known_labels[match] if dists[match] < threshold else "Unknown"

            # Scale back up
            top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- GUI ----------
def clear_form():
    for widget in form_widgets:
        widget.destroy()
    form_widgets.clear()

def show_form():
    clear_form()

    label_name = tk.Label(root, text="Name:")
    entry = tk.Entry(root)
    label_name.pack(pady=3)
    entry.pack()
    form_widgets.extend([label_name, entry])
    global entry_name
    entry_name = entry

    label_age = tk.Label(root, text="Age:")
    age_entry = tk.Entry(root)
    label_age.pack(pady=3)
    age_entry.pack()
    form_widgets.extend([label_age, age_entry])
    global entry_age
    entry_age = age_entry

    label_gender = tk.Label(root, text="Gender:")
    label_gender.pack(pady=3)
    gender_frame = tk.Frame(root)
    gender_frame.pack()
    global gender_var
    gender_var = tk.StringVar(value="Male")
    genders = ["Male", "Female", "Other"]
    for g in genders:
        btn = tk.Radiobutton(gender_frame, text=g, variable=gender_var, value=g)
        btn.pack(side="left", padx=5)
        form_widgets.append(btn)
    form_widgets.extend([label_gender, gender_frame])

    confirm_btn = tk.Button(root, text="Confirm & Capture", command=on_capture_confirm)
    confirm_btn.pack(pady=10)
    form_widgets.append(confirm_btn)

def on_capture_confirm():
    name = entry_name.get().strip()
    age = entry_age.get().strip()
    gender = gender_var.get().strip()

    if not name or not age or not gender:
        messagebox.showerror("Input Error", "Please fill all fields.")
        return

    capture_faces_gui(name, age, gender)
    clear_form()

# ---------- Main GUI Window ----------
root = tk.Tk()
root.title("Face Registration & Recognition")
root.geometry("400x500")

tk.Label(root, text="Face Recognition System", font=("Arial", 16)).pack(pady=20)
tk.Button(root, text="Capture New Face", width=25, command=show_form).pack(pady=10)
tk.Button(root, text="Start Face Recognition", width=25, command=start_recognition).pack(pady=10)

root.mainloop()
