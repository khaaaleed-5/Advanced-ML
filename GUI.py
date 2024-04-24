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
         # Retrieve values from entry fields and variables
             df=save_user_inputs()
             encodded_df=encode_categorical_variables(df)
             print(encodded_df)
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
                latitude_value = float(self.latitude_entry.get())
                longitude_value = float(self.longitude_entry.get())
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
                        'latitude': latitude_value,
                        'longitude': longitude_value,
                        'baths': baths_value,
                        'bedrooms': bedrooms_value,
                        'property_type': property_type_value,
                        'purpose': purpose_value
                }
                
                # Convert the dictionary to a DataFrame
                df = pd.DataFrame(user_inputs_dict, index=[0])
                
                # Return the DataFrame
                return df
        
        # function that scales the user input data
        def z_score_scaling_manual(df):
                numeric_columns = ['baths', 'bedrooms', 'Area Size', 'latitude', 'longitude']  # Get numeric column names

                
                scaled_data_array = []  # Empty list to store scaled data for each column
                
                Dataset=handling_Datset(False)
                
                for column in numeric_columns:
                        # Calculate mean and standard deviation for the current column
                        mean = np.mean(Dataset[column])
                        std_dev = np.std(Dataset[column])
                        
                        
                        #Z-score scaling formula: (x - mean) / std_dev
                        scaled_column = (df[column] - mean) / std_dev
                        
                        #Append scaled column to the array
                        scaled_data_array.append(scaled_column)
                
                 # Convert the list of scaled columns into a DataFrame
                scaled_df = pd.DataFrame(scaled_data_array).T   # Transpose the DataFrame to match original DataFrame shape
                #scaled_df.columns = numeric_columns  # Set column names
                
                return scaled_df

        
        
        def encode_categorical_variables(User_Data_Df):
                        # Creating a dictionary with variable names as keys and their values as lists
                
                        
                        # Load your dataset
                        dataset = handling_Datset(True)  # Assuming this function loads your dataset
                        categorical_columns = ['property_type', 'city', 'province_name', 'purpose', 'location', 'agent', 'agency']
                        
                        # Create a new DataFrame with only the specified columns to encode
                        uDF=User_Data_Df.copy()
                        
                        uDF = uDF[categorical_columns]
                        
        
                        # Fit the encoder on the dataset and transform the provided data
                        encoder = ce.TargetEncoder(cols=categorical_columns)
                        encoder.fit(dataset[categorical_columns], dataset['price'])
                        encoded_df = encoder.transform(uDF)
                        
                        unscaled_df=User_Data_Df.drop(columns=categorical_columns)
                        scaled_df=z_score_scaling_manual(unscaled_df)
                        
                        # Combine the encoded columns with the rest of the original DataFrame
                        df_encoded = pd.concat([scaled_df, encoded_df], axis=1)
                        
                        return df_encoded
            


        def preprocess_dataset(dataset):
                df = encode_categorical_variables(df)
                return df

                
        def handling_Datset(scale=False):
                dataset = pd.read_csv('./Datasets/regression.csv')
                
                # droping duplictes       
                drop_colums=['property_id', 'location_id', 'page_url']
                df = dataset.drop(columns= drop_colums, axis=1)
                df = df.drop_duplicates()
                # droping na values
                df = df.dropna()
                # handling outliers
                numeric_columns = ['price', 'baths', 'Area Size', 'bedrooms']
                new_df=df.copy()
                for col in numeric_columns:
                    Q1 = new_df[col].quantile(0.25)
                    Q3 = new_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    new_df[col] = np.where(
                        new_df[col] > upper_bound,
                        upper_bound,
                        np.where(
                            new_df[col] < lower_bound,
                            lower_bound,
                            new_df[col]))
                df = new_df   
                
                scaler = StandardScaler()

                # Scaling our numeric data
                if scale==True: 
                        df['price']=scaler.fit_transform(df[['price']])
                        return df
                else:
                        return df

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

# Latitude
        latitude_frame = tk.Frame(self, bg='#041618')
        latitude_frame.pack(pady=5)
        latitude_label = tk.Label(latitude_frame, text="Latitude:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        latitude_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.latitude_entry = tk.Entry(latitude_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.latitude_entry.grid(row=0, column=1, padx=10, pady=5)

# Longitude

        longitude_frame = tk.Frame(self, bg='#041618')
        longitude_frame.pack(pady=5)
        longitude_label = tk.Label(longitude_frame, text="Longitude:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        longitude_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.longitude_entry = tk.Entry(longitude_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.longitude_entry.grid(row=0, column=1, padx=10, pady=5)

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
        svr_model = load_model('./SVM/models/svrmodel.pkl')
        return svr_model

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
