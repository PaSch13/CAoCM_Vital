import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import os

def submit_data():
    age = age_entry.get()
    height = height_entry.get()
    sex = sex_entry.get()
    bmi = bmi_entry.get()

    # Generate a random number between 0 and 100 for complication possibility
    complication_possibility = random.randint(0, 100)
    complication_label.config(text=f"{complication_possibility} %")
    
    # Set color based on value
    if complication_possibility < 25:
        complication_label.config(fg="green")
    elif complication_possibility < 50:
        complication_label.config(fg="yellow")
    else:
        complication_label.config(fg="red")

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Here you can add code to process the uploaded file
        messagebox.showinfo("File Uploaded", f"File uploaded: {file_path}")

# Create the main window
root = tk.Tk()
root.title("Patient Data Entry")

# Configure the grid to be resizable
for i in range(8):
    root.grid_rowconfigure(i, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=2)

# Age
tk.Label(root, text="Age:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Height
tk.Label(root, text="Height (cm):").grid(row=1, column=0, padx=10, pady=10, sticky="e")
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# Sex
tk.Label(root, text="Sex:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
sex_entry = tk.Entry(root)
sex_entry.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

# BMI
tk.Label(root, text="BMI:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
bmi_entry = tk.Entry(root)
bmi_entry.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

# Submit Button
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="nsew")

# Upload File Button
upload_button = tk.Button(root, text="Upload Patient Data File", command=upload_file)
upload_button.grid(row=5, column=0, columnspan=2, pady=10, sticky="nsew")

# Estimated Complication Possibility
complication_frame = tk.Frame(root, bd=2, relief="groove")
complication_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=10, sticky="nsew")

complication_headline = tk.Label(complication_frame, text="Estimated Complication Possibility", font=("Arial", 14))
complication_headline.pack(pady=10)

complication_label = tk.Label(complication_frame, text="0 %", font=("Arial", 24), width=10, height=5)
complication_label.pack(expand=True)

# Further Details
details_frame = tk.Frame(root, bd=2, relief="groove")
details_frame.grid(row=3, column=2, rowspan=3, padx=10, pady=10, sticky="nsew")

details_headline = tk.Label(details_frame, text="Further Details", font=("Arial", 14))
details_headline.pack(pady=10)

expected_icu_days_label = tk.Label(details_frame, text="Expected ICU Days:", font=("Arial", 12))
expected_icu_days_label.pack(pady=5)

expected_icu_days_value = tk.Label(details_frame, text="0", font=("Arial", 12))
expected_icu_days_value.pack(pady=5)

second_detail_label = tk.Label(details_frame, text="Second Detail:", font=("Arial", 12))
second_detail_label.pack(pady=5)

second_detail_value = tk.Label(details_frame, text="Detail Value", font=("Arial", 12))
second_detail_value.pack(pady=5)

# Get the directory of this script
script_dir = os.path.dirname(__file__)

# Load and resize images using Pillow with relative paths
image1_path = os.path.join(script_dir, "images/miti_logo.png")
image2_path = os.path.join(script_dir, "images/tum_logo.png")

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Resize images to a common size
new_size = (50, 40)  # You can adjust the size as needed
image1 = image1.resize(new_size, Image.LANCZOS)
image2 = image2.resize(new_size, Image.LANCZOS)

# Convert images to PhotoImage
image1 = ImageTk.PhotoImage(image1)
image2 = ImageTk.PhotoImage(image2)

# Add image labels at the bottom on the right side, next to each other
image_label1 = tk.Label(root, image=image1)
image_label1.grid(row=6, column=2, sticky="e", padx=60, pady=5)

image_label2 = tk.Label(root, image=image2)
image_label2.grid(row=6, column=2, sticky="e", padx=5, pady=5)

# Run the application
root.mainloop()