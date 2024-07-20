import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def submit_data():
    age = age_entry.get()
    height = height_entry.get()
    sex = sex_entry.get()
    bmi = bmi_entry.get()

    # Here you can add code to process the patient data
    messagebox.showinfo("Patient Data", f"Age: {age}\nHeight: {height}\nSex: {sex}\nBMI: {bmi}")

def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Here you can add code to process the uploaded file
        messagebox.showinfo("File Uploaded", f"File uploaded: {file_path}")

# Create the main window
root = tk.Tk()
root.title("Patient Data Entry")

# Age
tk.Label(root, text="Age:").grid(row=0, column=0, padx=10, pady=10)
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1, padx=10, pady=10)

# Height
tk.Label(root, text="Height (cm):").grid(row=1, column=0, padx=10, pady=10)
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1, padx=10, pady=10)

# Sex
tk.Label(root, text="Sex:").grid(row=2, column=0, padx=10, pady=10)
sex_entry = tk.Entry(root)
sex_entry.grid(row=2, column=1, padx=10, pady=10)

# BMI
tk.Label(root, text="BMI:").grid(row=3, column=0, padx=10, pady=10)
bmi_entry = tk.Entry(root)
bmi_entry.grid(row=3, column=1, padx=10, pady=10)

# Submit Button
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.grid(row=4, column=0, columnspan=2, pady=10)

# Upload File Button
upload_button = tk.Button(root, text="Upload Patient Data File", command=upload_file)
upload_button.grid(row=5, column=0, columnspan=2, pady=10)

# Run the application
root.mainloop()