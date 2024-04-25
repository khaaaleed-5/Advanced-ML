import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import category_encoders as ce
import joblib
import pickle
import os



abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SVM/')


#All pages
class Page(tk.Frame):
    def __init__(self, parent, controller, title):
        tk.Frame.__init__(self, parent, bg='#041618')
        self.controller = controller
        
        label = tk.Label(self, text=title, bg='#041618', fg="#FFFFC7", font=("Helvetica", 32), width=20, height=2)
        label.pack(pady=0, padx=10)

        if not isinstance(self, StartPage):
            button_back = tk.Button(self, text="Back", width=10, height=1, bg='#FFFFC7', command=lambda: controller.show_frame(StartPage))
            button_back.pack(pady=5, padx=10)

#home page
class StartPage(Page):
    def __init__(self, parent, controller):
        Page.__init__(self, parent, controller, "Choose Model")

        button1 = tk.Button(self, text="ANN", width=20, height=2, bg='#FFFFC7', command=lambda: controller.show_frame(PageOne))
        button1.pack(pady=10, padx=10)

        button2 = tk.Button(self, text="SVM", width=20, height=2, bg='#FFFFC7', command=lambda: controller.show_frame(PageTwo))
        button2.pack(pady=10, padx=10)

        button3 = tk.Button(self, text="Decision tree", width=20, height=2, bg='#FFFFC7', command=lambda: controller.show_frame(PageThree))
        button3.pack(pady=10, padx=10)

#ANN page
class PageOne(Page):
    def __init__(self, parent, controller):
        Page.__init__(self, parent, controller, "ANN")

        # Create a button to upload a photo
        upload_button = tk.Button(self, text="Upload Photo", width=10, height=1, bg='#FFFFC7', command=self.upload_photo)
        upload_button.pack(pady=10, padx=10)
        
        # Create a label to display the uploaded photo
        self.label = tk.Label(self)
        self.label.pack(padx=10, pady=10)

        # Create a label to display the prediction result
        self.prediction_label = tk.Label(self, text="", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

    def upload_photo(self):
        filename = filedialog.askopenfilename()
        labels = ['fresh','rotten']
        def preprocess_fun(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(28,28))
            img = img.astype(np.float32) / 255.0
            return img
        
        if filename:
            img = preprocess_fun(filename)
            arr = []
            arr.append(img)
            arr = np.array(arr)
            ANN_model = self.controller.load_ANN_model()
            pred = ANN_model.predict(arr)
            pred_class = np.argmax(pred)
            pred_label = labels[pred_class]
            
            # Open the selected image file
            image = Image.open(filename)
            # Resize the image to fit within a certain size (e.g., 256x256)
            image = image.resize((256, 256), Image.ANTIALIAS)
            # Convert the image for tkinter
            photo = ImageTk.PhotoImage(image)
            # Display the image on the label
            self.label.config(image=photo)
            self.label.image = photo
            
            # Update the prediction label with the predicted class
            self.prediction_label.config(text=f"Prediction: {pred_label}")


# SVM page
class PageTwo(Page):
    def __init__(self, parent, controller):
        Page.__init__(self, parent, controller, "SVM")
        
        def process_inputs():
             desired_columns=['property_type' , 'city' , 'province_name', 'purpose'  ,'location' ,'agent' , 'agency' , 'baths'  ,  'bedrooms','Area Size' ]

             # Retrieve values from entry fields and variables
             df=encode_categorical_variables()
             df = df.reindex(columns=desired_columns)
             svr_model=self.controller.load_SVM_model()
             pred=svr_model.predict(df)
             pred = pred.reshape(1, -1)

             with open(abspath+"target_scaler.pkl","rb") as f:
                        scaler=pickle.load(f)
             org_pred=scaler.inverse_transform(pred)
             # Update the prediction label with the predicted class
             self.prediction_label.config(text=f"Prediction: {int(org_pred[0][0])}")
             
            
             
             

             
             #z_score_scaling_manual(df)
            

            
    # function that read the user inputs
    
        def save_user_inputs():
                global user_inputs_df
                
                # Retrieve values from entry fields and variables
                agency_value = self.agency_entry.get()
                agent_value = self.agent_entry.get()
                area_value = float(self.area_entry.get())
                location_value = self.location_entry.get()
                city_value = self.city_entry.get()
                province_name_value = self.province_name_entry.get()
                baths_value = self.baths_var.get()
                bedrooms_value = self.bedrooms_var.get()
                property_type_value = self.property_type_var.get()
                purpose_value = self.purpose_var.get()
                
                # Create a dictionary with user inputs
                user_inputs_dict = {
                        'agency': agency_value,
                        'agent': agent_value,
                        'Area Size': area_value,
                        'location': location_value,
                        'city': city_value,
                        'province_name': province_name_value,
                        'baths': baths_value,
                        'bedrooms': bedrooms_value,
                        'property_type': property_type_value,
                        'purpose': purpose_value
                }
                
                # Convert the dictionary to a DataFrame
                df = pd.DataFrame(user_inputs_dict, index=[0])
                
                # Return the DataFrame
                return df
        
    
        
        def encode_categorical_variables():
                
                        # # Creating a dictionary with variable names as keys and their values as lists
                        
                with open(abspath+"encoder.pkl","rb") as f :
                    encoder = pickle.load(f)
                                 
                df=scaling_data()
                df2=df.copy()
                df2=df2.drop(columns=['agency', 'agent', 'location', 'city', 'province_name', 'property_type','purpose'])
                df=encoder.transform(df[['agency', 'agent', 'location', 'city', 'province_name', 'property_type','purpose']])
                df = pd.concat([df2, df], axis=1)

                return df
                                 
                                 
                 
        def scaling_data():
                
                with open(abspath+"scaler.pkl","rb") as f:
                        scaler=pickle.load(f)
                        
                df=save_user_inputs()
                        
                df[['baths', 'bedrooms', 'Area Size']]=scaler.transform(df[['baths', 'bedrooms', 'Area Size']])
                        
                return df
        #        

# Agency
        agency_frame = tk.Frame(self, bg='#041618')
        agency_frame.pack(pady=5)
        agency_label = tk.Label(agency_frame, text="Agency:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        agency_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.agency_entry = tk.Entry(agency_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.agency_entry.grid(row=0, column=1, padx=10, pady=5)

# Agent
        agent_frame = tk.Frame(self, bg='#041618')
        agent_frame.pack(pady=5)
        agent_label = tk.Label(agent_frame, text="Agent:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        agent_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.agent_entry = tk.Entry(agent_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.agent_entry.grid(row=0, column=1, padx=10, pady=5)

# Area Size
        area_frame = tk.Frame(self, bg='#041618')
        area_frame.pack(pady=5)
        area_label = tk.Label(area_frame, text="Area Size:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        area_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.area_entry = tk.Entry(area_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.area_entry.grid(row=0, column=1, padx=10, pady=5)

# Location
        location_frame = tk.Frame(self, bg='#041618')
        location_frame.pack(pady=5)
        location_label = tk.Label(location_frame, text="Location:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        location_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.location_entry = tk.Entry(location_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.location_entry.grid(row=0, column=1, padx=10, pady=5)

# City
        city_frame = tk.Frame(self, bg='#041618')
        city_frame.pack(pady=5)
        city_label = tk.Label(city_frame, text="City:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        city_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.city_entry = tk.Entry(city_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.city_entry.grid(row=0, column=1, padx=10, pady=5)

# Province Name
        province_name_frame = tk.Frame(self, bg='#041618')
        province_name_frame.pack(pady=5)
        province_name_label = tk.Label(province_name_frame, text="Province Name:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        province_name_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.province_name_entry = tk.Entry(province_name_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.province_name_entry.grid(row=0, column=1, padx=10, pady=5)


# Baths
        baths_frame = tk.Frame(self, bg='#041618')
        baths_frame.pack(pady=5)
        baths_label = tk.Label(baths_frame, text="Baths:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        baths_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.baths_var = tk.IntVar(self)
        self.baths_var.set(0)
        baths_spinbox = tk.Spinbox(baths_frame, from_=0, to=10, increment=1, textvariable=self.baths_var, font=("Helvetica", 12))
        baths_spinbox.grid(row=0, column=1, padx=10, pady=5)

# Bedrooms
        bedrooms_frame = tk.Frame(self, bg='#041618')
        bedrooms_frame.pack(pady=5)
        bedrooms_label = tk.Label(bedrooms_frame, text="Bedrooms:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        bedrooms_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.bedrooms_var = tk.IntVar(self)
        self.bedrooms_var.set(0)
        bedrooms_spinbox = tk.Spinbox(bedrooms_frame, from_=0, to=10, increment=1, textvariable=self.bedrooms_var, font=("Helvetica", 12))
        bedrooms_spinbox.grid(row=0, column=1, padx=10, pady=5)

# Property Type
        property_type_frame = tk.Frame(self, bg='#041618')
        property_type_frame.pack(pady=5)
        property_type_label = tk.Label(property_type_frame, text="Property Type:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        property_type_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.property_type_var = tk.StringVar(self)
        self.property_type_var.set("House")
        property_type_menu = tk.OptionMenu(property_type_frame, self.property_type_var, "House", "Flat", "Farm House", "Room", "Upper Portion", "Lower Portion", "Penthouse")
        property_type_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        property_type_menu.grid(row=0, column=1, padx=10, pady=5)

# Purpose
        purpose_frame = tk.Frame(self, bg='#041618')
        purpose_frame.pack(pady=5)
        purpose_label = tk.Label(purpose_frame, text="Purpose:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        purpose_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.purpose_var = tk.StringVar(self)
        self.purpose_var.set("For Sale")
        purpose_menu = tk.OptionMenu(purpose_frame, self.purpose_var, "For Sale", "For Rent")
        purpose_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        purpose_menu.grid(row=0, column=1, padx=10, pady=5)
        
# Create a label to display the prediction result
        self.prediction_label = tk.Label(self, text="", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        
# Create a button to trigger the function
        process_button = tk.Button(self, text="Process Inputs", command=process_inputs, bg='#FFFFC7', font=("Helvetica", 14))
        process_button.pack(pady=10)
        
        



#Decision tree page
class PageThree(Page):
    def __init__(self, parent, controller):
        Page.__init__(self, parent, controller, "Decision tree")

class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('ADV-ML')
        self.configure(bg='#041618')
        self.minsize(width=1000,height=700)
        self.maxsize(width=1920,height=1080)

        container = tk.Frame(self, bg='#041618')
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
    
    #load models 
    def load_ANN_model(self):
        ANN_model = load_model('./NN/model.h5')
        return ANN_model
    def load_SVM_model(self):
        svr_model = joblib.load('./SVM/models/svrmodel.pkl')
        return svr_model

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
