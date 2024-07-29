import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

selected_file_path = ""
selected_patient_data_path = ""

def submit_data():
    # Get input data
    age = age_entry[0].get()
    sex = sex_entry[0].get()
    height = height_entry[0].get()
    weight = weight_entry[0].get()
    asa = asa_entry[0].get()
    emop = emop_entry[0].get()
    opttype = opttype_entry[0].get()


    bmi = float(weight) / ((float(height) / 100) ** 2)


    # Prepare the data list
    data_list = [age, sex, height, weight, bmi, asa, emop, opttype]

    # Handle missing values
    for idx, entry in enumerate(data_list):
        if entry == "":
            data_list[idx] = 0

    # Convert sex to binary
    if sex.lower() in ['male', 'm']:
        data_list[1] = 1
    else:
        data_list[1] = 0

    # Define the operation type dictionary
    optype_dict = {
        'colorectal': 0,
        'stomach': 1,
        'biliary/pancreas': 2,
        'vascular': 3,
        'major resection': 4,
        'breast': 5,
        'minor resection': 6,
        'transplantation': 7,
        'hepatic': 8,
        'thyroid': 9,
        'others': 10,
        '': 11
    }

    # Map operation type to number
    if opttype.lower() in optype_dict:
        data_list[-1] = optype_dict[opttype.lower()]
    else:
        raise ValueError(f"Operation type '{opttype}' is not recognized.")

    # Convert all elements to float and ensure dtype is float64
    data_list = np.array([float(i) for i in data_list], dtype=np.float64)

    # Ensure the data shape is correct
    if data_list.shape != (8,):
        raise ValueError(f"Input data must have shape (8,), but got shape {data_list.shape}")

    # Load the models
    model_comp_proba = joblib.load('/Users/patrickschneider/Desktop/CAoCM/CAoCM_Vital/CaoCom_Model_StaticData/model_classifier/gradient_boosting_model.pkl')
    model_icu_days = tf.keras.models.load_model('/Users/patrickschneider/Desktop/CAoCM/CAoCM_Vital/CaoCom_Model_StaticData/model_predictor/icu_predictor_model.h5')

    # # Recompile the ICU days model if necessary
    # model_icu_days.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    scaler_loaded = joblib.load('CaoCom_Model_StaticData/model_classifier/scaler.pkl')
    data_list_reduced = scaler_loaded.transform([data_list[:-1]])  # Ensure data is in a 2D array  

    # Make complication prediction
    complication_possibility = model_comp_proba.predict_proba(data_list_reduced[0].reshape(1, -1))[0][1]*100

    # Set color based on value
    if complication_possibility < 25:
        fg="green"
    elif complication_possibility < 50:
        fg="yellow"
    else:
        fg="red"

    # Update the complication label
    complication_label.config(text=f"{complication_possibility:.1f} %", fg=fg)

    # # Preprocess the new data
    # scaler = StandardScaler()
    # df_new_scaled = scaler.fit_transform(data_list.reshape(1, -1))

    # ICU prediction
    icu_days_predictions = model_icu_days.predict(data_list.reshape(1, -1)).flatten()[0] #TODO: check prediction/df_new_scaled
    
    # update the expected ICU days label
    expected_icu_days_label.config(text=f"{icu_days_predictions:.1f}")


def submit_data_files():
    global selected_file_path
    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(selected_file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Choose a valid CSV file: {e}")
        return
    
    #TODO: Call classifier from Michael and return complication possibility

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
    global selected_file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_file_path = file_path
        file_name = os.path.basename(file_path)
        selected_file_entry.config(state='normal')
        selected_file_entry.delete(0, tk.END)
        selected_file_entry.insert(0, file_name)
        selected_file_entry.config(state='readonly')
        # messagebox.showinfo("File Uploaded", f"File uploaded: {file_path}")

def upload_patient_data(layout1_widgets):
    global selected_patient_data_path
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])

    if not file_path:
        return  # If no file was selected, return early

    try:
        # Read the Excel file, treating the first row as data (not as header)
        df = pd.read_excel(file_path, header=None)
    except Exception as e:
        messagebox.showerror("Error", f"Choose a valid Excel file: {e}")
        return

    # Ensure the second last column's data is numeric
    second_last_col_idx = -2
    df.iloc[:, second_last_col_idx] = pd.to_numeric(df.iloc[:, second_last_col_idx], errors='coerce').fillna(0).astype(int)

    if file_path:
        selected_patient_data_path = file_path
        for idx, widget in enumerate(layout1_widgets[:-1]):
            if idx < len(df.columns):
                widget[0].delete(0, tk.END)
                widget[0].insert(0, df.values[0][idx])  # Assuming you want to insert the first row of the dataframe
            else:
                widget[0].insert(0, "")  # In case there are more widgets than columns in the dataframe

def change_layout(event):
    selected_layout = layout_var.get()
    if selected_layout == "Layout 1":
        show_layout1()
    elif selected_layout == "Layout 2":
        show_layout2()

def show_layout1():
    # Show all elements for layout 1
    for widget in layout2_widgets:
        widget[0].grid_remove()
    for widget in layout2_widgets:
        widget[1].grid_remove()
    for widget in layout1_widgets:
        widget[0].grid()
    for widget in layout1_widgets:
        widget[1].grid()
    upload_button.grid(row=8, column=2, columnspan=2, pady=entry_pady, padx=(10, 0), sticky="nsew")

def show_layout2():
    # Hide all elements for layout 1 except submit button, add upload buttons
    for widget in layout1_widgets:
        widget[0].grid_remove()
    for widget in layout1_widgets:
        widget[1].grid_remove()
    for widget in layout2_widgets:
        widget[0].grid()
    for widget in layout2_widgets:
        widget[1].grid()
    upload_button.grid_remove()
    selected_file_label.grid(row=2, column=0, padx=entry_padx, pady=entry_pady, sticky="e")
    selected_file_entry.grid(row=2, column=1, padx=entry_padx, pady=entry_pady, ipady=entry_ipady, sticky="nsew")

# Create the main window
root = tk.Tk()
root.title("Comlication and ICU Prediction")

# Configure the grid to be resizable
for i in range(10):
    root.grid_rowconfigure(i, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=0)  # Ensure column 2 doesn't expand
root.grid_columnconfigure(3, weight=1)  # Add a dummy column to the right to fill∂ space

# Layout Selection Dropdown
layout_var = tk.StringVar(value="General")
layout_dropdown = tk.OptionMenu(root, layout_var, "General", "Layout 2", command=change_layout)
label = tk.Label(root, text="Algorithm:")
label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
layout_dropdown.grid(row=0, column=1, padx=10, pady=10, ipady=5, sticky="nsew")
layout_dropdown.config(fg="black")  # Change font color to black

# Access the menu of the OptionMenu to change its item colors
menu = root.nametowidget(layout_dropdown.menuname)
menu.config(fg="black")  # Change font color of menu items to black

# Uniform padding and size configuration for entries
entry_padx = 10
entry_pady = 10
entry_ipady = 5

# Helper function to create labels and entries
def create_label_entry(row, text):
    label = tk.Label(root, text=text)
    label.grid(row=row, column=0, padx=entry_padx, pady=entry_pady, sticky="e")
    entry = tk.Entry(root)
    entry.grid(row=row, column=1, padx=entry_padx, pady=entry_pady, ipady=entry_ipady, sticky="nsew")
    return [entry,label]

# Age
age_entry = create_label_entry(1, "Age:")

# Sex
sex_entry = create_label_entry(2, "Sex:")

# Height
height_entry = create_label_entry(3, "Height (cm):")

# Weight
weight_entry = create_label_entry(4, "Weight (kg):")

# ASA
asa_entry = create_label_entry(5, "ASA:")

# EMOP
emop_entry = create_label_entry(6, "EMOP:")

# Operation Type
opttype_entry = create_label_entry(7, "Operation Type:")

# Submit Button
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.grid(row=8, column=1, columnspan=1, pady=entry_pady, padx=entry_padx, sticky="nsew")

# Widgets for layout 1
layout1_widgets = [age_entry, sex_entry, height_entry, weight_entry, asa_entry, emop_entry, opttype_entry, [submit_button, submit_button]]

# Submit Files Button
submit_button_files = tk.Button(root, text="Submit Files", command=submit_data_files)
submit_button_files.grid(row=8, column=1, columnspan=1, pady=entry_pady, padx=entry_padx, ipady=entry_ipady, sticky="nsew")

# Upload File Button (to be removed in Layout 2)
upload_button = tk.Button(root, text="Upload Patient Data File", command=lambda: upload_patient_data(layout1_widgets))
upload_button.grid(row=8, column=2, pady=entry_pady, padx=(10, 0), sticky="ew")

# Upload File Buttons for Layout 2
label_upload_file_button1 = tk.Label(root, text="Patient Data File 1")
label_upload_file_button1.grid(row=1, column=0, padx=entry_padx, pady=entry_pady, sticky="e")
upload_file_button1 = tk.Button(root, text="File 1", command=upload_file)

# Label and Entry for displaying selected file name
selected_file_label = tk.Label(root, text="Selected file:")
selected_file_entry = tk.Entry(root, state='readonly', bd=0, highlightthickness=0)

# Estimated Complication Possibility
complication_frame = tk.Frame(root, bd=2)
complication_frame.grid(row=1, column=2, rowspan=2, padx=entry_padx, pady=entry_pady, sticky="nsew")

complication_headline = tk.Label(complication_frame, text="Estimated Complication Possibility", font=("Arial", 18))
complication_headline.pack(pady=10)

complication_label = tk.Label(complication_frame, text="0 %", font=("Arial", 34), width=0, height=0)
complication_label.pack(expand=True)

# Further Details
details_frame = tk.Frame(root, bd=2)
details_frame.grid(row=4, column=2, rowspan=3, padx=entry_padx, pady=entry_pady, sticky="nsew")

details_headline = tk.Label(details_frame, text="Expected ICU Days:", font=("Arial", 18))
details_headline.pack(pady=10)

expected_icu_days_label = tk.Label(details_frame, text="0", font=("Arial", 18))
expected_icu_days_label.pack(pady=5)

# Widgets for layout 2
layout2_widgets = [
    [upload_file_button1, label_upload_file_button1],
    [submit_button_files, submit_button_files],
    [selected_file_entry, selected_file_label]
]

upload_file_button1.grid(row=1, column=1, pady=entry_pady, padx=entry_padx, sticky="nsew")

# Hide layout 2 widgets initially
for widget in layout2_widgets:
    widget[0].grid_remove()
    widget[1].grid_remove()

# Show layout 1 by default
show_layout1()

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
image_label1.grid(row=9, column=2, sticky="e", padx=60, pady=5)

image_label2 = tk.Label(root, image=image2)
image_label2.grid(row=9, column=2, sticky="e", padx=5, pady=5)

# Run the application
root.mainloop()