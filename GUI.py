import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
import joblib
import pickle
import os

# Saving the abs Path so we can use it to call Both the Scalers and the Encoders we Saved
abspath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SVM/')


dtpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Decision Tree/')

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
        
        # Function that makes predictions and displays it 
        def predict_output():
                
             # Cols with the order of the features trained on the model   
             reindexed_columns=['property_type' , 'city' , 'province_name', 'purpose'  ,'location' ,'agent' , 'agency' , 'baths'  ,  'bedrooms','Area Size' ]
             
             # Saving the user inputs data into a dataframe
             df=save_user_inputs()
             
             # Scaling the user input data
             df=scaling_data(df)
             
             # Encoding the user input data
             df=encode_categorical_variables(df)
             
             # Reindexing the user inputs dataframe to fit the order of the model order ( the one that was trained on )   
             df = df.reindex(columns=reindexed_columns)
             
             # Using the loaded model to Predict the price 
             svr_model=self.controller.load_SVM_model()
             pred=svr_model.predict(df)
             pred = pred.reshape(1, -1)
             
             # Calling the loaded target scaler for inverse scaling our prediction
             with open(abspath+"target_scaler.pkl","rb") as f:
                        scaler=pickle.load(f)
             org_pred=scaler.inverse_transform(pred)
             
             # Updating the prediction label with the predicted class
             self.prediction_label.config(text=f"Prediction: {int(org_pred[0][0])}")
        
        # Function that does the math on the area size to convert it from Marla or Kanal to Square Meter
        def handle_area_Size():
                # Reading the values from the user input fields
                area_value = float(self.area_size.get())
                area_type = self.area_type.get()
                
                # converting them from Marla to square meter
                if  area_type=="Marla":
                   area_value *= 25.2929
                # converting them from Kanal to square meter
                else:
                    area_value *= 505.857   
                    
                # returning the value 
                return area_value      

              
        # function that read the user inputs      
        def save_user_inputs():
                            
                # calling the math function to preprocess the area size field
                area_size = handle_area_Size()
                
                # Retrieve values from entry fields and variables
                agency_value = self.agency_entry.get()
                agent_value = self.agent_entry.get()
                location_value = self.location_entry.get()
                city_value = self.city_var.get()
                province_name_value = self.province_name_var.get()
                baths_value = self.baths_var.get()
                bedrooms_value = self.bedrooms_var.get()
                property_type_value = self.property_type_var.get()
                purpose_value = self.purpose_var.get()
                
                # Create a dictionary with user inputs
                user_inputs_dict = {
                        'agency': agency_value,
                        'agent': agent_value,
                        'Area Size':area_size,
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
        
    
        # encoding the user input data
        def encode_categorical_variables(df):
                
                # loading the Encoder we need        
                with open(abspath+"encoder.pkl","rb") as f :
                    encoder = pickle.load(f)
                    
                # Copying the dataframe of user input data we need so we can perform encoding only on the categorial data                 
                df2=df.copy()
                
                # Encoding the categorial data in the dataframe we need
                df=encoder.transform(df[['agency', 'agent', 'location', 'city', 'province_name', 'property_type','purpose']])
                
                # Droping the categorial data from the dataframe that is not encoded
                df2=df2.drop(columns=['agency', 'agent', 'location', 'city', 'province_name', 'property_type','purpose'])
                
                # Merging the numeric dataframe with the encoded numreic one
                df = pd.concat([df2, df], axis=1)

                # Returning the dataframe after encoding
                return df
                                 
                                 
        # Scaling the user input data         
        def scaling_data(df):
                # Loading the Scaler we need 
                with open(abspath+"scaler.pkl","rb") as f:
                        scaler=pickle.load(f)
                        
                # Scaling the dataframe which has the User Input data        
                df[['baths', 'bedrooms', 'Area Size']]=scaler.transform(df[['baths', 'bedrooms', 'Area Size']])
                    
                # Returning the dataframe after Scaling data        
                return df
            
            
             # Function to randomize user inputs according to some examples we put earlier ( just to save time during the model reveal)
        def random_testing_values():
            
                # Reading the test values 
                dt = pd.read_csv('./SVM/test_values/Book1.csv')
                  
                # Saving them into array      
                property_type_values = np.ravel(dt['property_type'])
                location_values = np.ravel(dt['location'])
                city_values = np.ravel(dt['city'])
                province_name_values = np.ravel(dt['province_name'])
                baths_values = np.ravel(dt['baths'])
                purpose_values = np.ravel(dt['purpose'])
                bedrooms_values = np.ravel(dt['bedrooms'])
                agency_values = np.ravel(dt['agency'])
                agent_values = np.ravel(dt['agent'])
                area_type_values = np.ravel(dt['Area Type'])
                area_Size_values = np.ravel(dt['Area Size'])

                # Picking a random number in 0 to 6 since we only have 7 test examples
                index = np.random.randint(0, 6)
            
                # Clearing the agency field and setting the random test value to it ( one of the ones we have earlier)
                self.agency_entry.delete(0, tk.END)
                self.agency_entry.insert(0, agency_values[index])

                # Clearing the agent field and setting the random test value to it ( one of the ones we have earlier)
                self.agent_entry.delete(0, tk.END)
                self.agent_entry.insert(0, agent_values[index])
                
                # Clearing the location field and setting the random test value to it ( one of the ones we have earlier)
                self.location_entry.delete(0, tk.END)
                self.location_entry.insert(0, location_values[index])
                
                # City field , setting the random test value to it ( one of the ones we have earlier)
                self.city_var.set(city_values[index])
                
                # Province name field , setting the random test value to it ( one of the ones we have earlier)
                self.province_name_var.set(province_name_values[index])
                
                # Baths field , setting the random test value to it ( one of the ones we have earlier)    
                self.baths_var.set(baths_values[index])
                
                # Bedrooms field , setting the random test value to it ( one of the ones we have earlier)
                self.bedrooms_var.set(bedrooms_values[index])
                
                # Property type field , setting the random test value to it ( one of the ones we have earlier)
                self.property_type_var.set(property_type_values[index])
                
                # Purpose field , setting the random test value to it ( one of the ones we have earlier)
                self.purpose_var.set(purpose_values[index])
                
                # Clearing the area Type field and setting the random test value to it 
                self.area_type.delete(0, "end")
                self.area_type.set(area_type_values[index])
                
                # Clearing the area Size field and setting the random test value to it 
                self.area_size.delete(0, "end")
                self.area_size.insert(0,area_Size_values[index])
                
                    
                       
            
                

        # Agency field
        agency_frame = tk.Frame(self, bg='#041618')
        agency_frame.pack(pady=5)
        agency_label = tk.Label(agency_frame, text="Agency:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        agency_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.agency_entry = tk.Entry(agency_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.agency_entry.grid(row=0, column=1, padx=10, pady=5)

        # Agent field
        agent_frame = tk.Frame(self, bg='#041618')
        agent_frame.pack(pady=5)
        agent_label = tk.Label(agent_frame, text="Agent:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        agent_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.agent_entry = tk.Entry(agent_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.agent_entry.grid(row=0, column=1, padx=10, pady=5)

         
        # Location field
        location_frame = tk.Frame(self, bg='#041618')
        location_frame.pack(pady=5)
        location_label = tk.Label(location_frame, text="Location:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        location_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.location_entry = tk.Entry(location_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.location_entry.grid(row=0, column=1, padx=10, pady=5)
        
        
        # Area Size 
        area_frame = tk.Frame(self, bg='#041618')
        area_frame.pack(pady=5)
        area_label = tk.Label(area_frame, text="Area Size:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 14))
        area_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.area_size = tk.Spinbox(area_frame, from_=0, to=1000, bg='#FFFFC7', font=("Helvetica", 10), width=7)
        self.area_size.grid(row=0, column=1, padx=(0, 5), pady=5)
        self.area_type = ttk.Combobox(area_frame, values=["marla", "kanal"], state="readonly", width=5)
        self.area_type.grid(row=0, column=2, padx=5, pady=5)
        self.area_type.current(0)  # Set default value to the first item

        # Baths field
        baths_frame = tk.Frame(self, bg='#041618')
        baths_frame.pack(pady=5)
        baths_label = tk.Label(baths_frame, text="Baths:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        baths_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.baths_var = tk.IntVar(self)
        self.baths_var.set(0)
        baths_spinbox = tk.Spinbox(baths_frame, from_=0, to=10, increment=1, textvariable=self.baths_var, font=("Helvetica", 12))
        baths_spinbox.grid(row=0, column=1, padx=10, pady=5)

        # Bedrooms field
        bedrooms_frame = tk.Frame(self, bg='#041618')
        bedrooms_frame.pack(pady=5)
        bedrooms_label = tk.Label(bedrooms_frame, text="Bedrooms:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        bedrooms_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.bedrooms_var = tk.IntVar(self)
        self.bedrooms_var.set(0)
        bedrooms_spinbox = tk.Spinbox(bedrooms_frame, from_=0, to=10, increment=1, textvariable=self.bedrooms_var, font=("Helvetica", 12))
        bedrooms_spinbox.grid(row=0, column=1, padx=10, pady=5)

        # Property Type field
        property_type_frame = tk.Frame(self, bg='#041618')
        property_type_frame.pack(pady=5)
        property_type_label = tk.Label(property_type_frame, text="Property Type:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        property_type_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.property_type_var = tk.StringVar(self)
        self.property_type_var.set("House")
        property_type_menu = tk.OptionMenu(property_type_frame, self.property_type_var, "House", "Flat", "Farm House", "Room", "Upper Portion", "Lower Portion", "Penthouse")
        property_type_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        property_type_menu.grid(row=0, column=1, padx=10, pady=5)

        
        # City field
        city_frame = tk.Frame(self, bg='#041618')
        city_frame.pack(pady=5)
        city_label = tk.Label(city_frame, text="City:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        city_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.city_var = tk.StringVar(self)
        self.city_var.set("islamabad")
        city_menu = tk.OptionMenu(city_frame, self.city_var, "islamabad", "Lahore","Karachi","Faisalabad","Rawalpindi")
        city_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        city_menu.grid(row=0, column=1, padx=10, pady=5)
        
        
        # Province Name field
        province_name_frame = tk.Frame(self, bg='#041618')
        province_name_frame.pack(pady=5)
        province_name_label = tk.Label(province_name_frame, text="Province Name:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        province_name_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.province_name_var = tk.StringVar(self)
        self.province_name_var.set("islamabad Capital")
        province_name_menu = tk.OptionMenu(province_name_frame, self.province_name_var, "islamabad Capital", "Punjab","Sindh")
        province_name_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        province_name_menu.grid(row=0, column=1, padx=10, pady=5)
        

        # Purpose field
        purpose_frame = tk.Frame(self, bg='#041618')
        purpose_frame.pack(pady=5)
        purpose_label = tk.Label(purpose_frame, text="Purpose:", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        purpose_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.purpose_var = tk.StringVar(self)
        self.purpose_var.set("For Sale")
        purpose_menu = tk.OptionMenu(purpose_frame, self.purpose_var, "For Sale", "For Rent")
        purpose_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        purpose_menu.grid(row=0, column=1, padx=10, pady=5)
        
        # Label to display the prediction result
        self.prediction_label = tk.Label(self, text="", bg='#041618', fg="#FFFFC7", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        
        # Button to set some random inputs 
        process_button = tk.Button(self, text="random test inputs", command=random_testing_values, bg='#FFFFC7', font=("Helvetica", 14))
        process_button.pack(pady=10)
        
        # Button to trigger the function
        process_button = tk.Button(self, text="Predict price", command=predict_output, bg='#FFFFC7', font=("Helvetica", 14))
        process_button.pack(pady=10)
        
        



#Decision tree page
class PageThree(Page): 
    def __init__(self, parent, controller):
        Page.__init__(self, parent, controller, "Decision tree")
        def save_user_inputs():
            global user_inputs_df
                    
            # Retrieve values from entry fields and variables
            age_value=self.age_entry.get()
            sex_value=self.Sex.get()
            chest_pain_type_value = self.chest_pain_type.get()
            resting_bps_value = self.resting_bps.get()
            cholesterol_value = self.cholesterol.get()
            fasting_blood_sugar_value = self.fasting_blood_sugar.get()
            resting_ecg_value = self.resting_ecg.get()
            max_heart_rate_value = self.max_heart_rate.get()
            exercise_angina_value = self.exercise_angina.get()
            oldpeak_value = self.oldpeak.get()
            ST_slope_value = self.ST_slope.get()

         #convert string value to intger 
         #sex
            if self.Sex.get() == "Male":
                        sex_value= 1
            elif self.Sex.get() == "Female":
                        sex_value= 0
            else:
                        return None  
           # chest pain type         
            if self.chest_pain_type.get() =="Typical angina":
                chest_pain_type_value = 1
            elif self.chest_pain_type.get() =="Atypical angina":
                chest_pain_type_value = 2
            elif self.chest_pain_type.get() =="Non-anginal pain":
                chest_pain_type_value = 3
            else:
                chest_pain_type_value = 4
            
            # Fasting blood sugar
            if self.fasting_blood_sugar.get()=='Fasting blood sugar > 120 mg/dl':
                 fasting_blood_sugar_value=1  
            else:
                 fasting_blood_sugar_value=0 
           #resting ecg         resting_ecg_menu = tk.OptionMenu(resting_ecg_frame, self.resting_ecg, "Normal", " Having ST-T wave abnormality","Showing probable or definite left ventricular hypertrophy by Estes criteria")

            if self.resting_ecg.get() =="Normal":
                resting_ecg_value  = 0
            elif self.resting_ecg.get() =="Having ST-T wave abnormality":
                resting_ecg_value  = 1
            else:
                resting_ecg_value  = 2
         # exercise angina
            if self.exercise_angina.get()=='YES':
                 exercise_angina_value=1  
            else:
                 exercise_angina_value=0
          #  ST slope         

            if self.ST_slope.get() == "Upsloping":
                    ST_slope_value = 1
            elif self.ST_slope.get() == "Flat":
                ST_slope_value = 2
            else:
                ST_slope_value = 3
         
            # Create a dictionary with user input data
            user_inputs_dict = {
                    'age': age_value,
                    'sex': sex_value,
                    'chest pain type': chest_pain_type_value,
                    'resting bp s': resting_bps_value,
                    'cholesterol': cholesterol_value,
                    'fasting blood sugar': fasting_blood_sugar_value,
                    'resting ecg': resting_ecg_value,
                    'max heart rate': max_heart_rate_value,
                    'exercise angina': exercise_angina_value,
                    'oldpeak': oldpeak_value,
                    'ST slope' : ST_slope_value
                    }
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(user_inputs_dict, index=[0])
            # Return the DataFrame
            return df
                  
        def scaling_data(df):
                
                with open(dtpath+"scaler.pkl","rb") as f:
                        scaler=pickle.load(f)
                        
                df=save_user_inputs()
                        
                df[['age','resting bp s','cholesterol','max heart rate','oldpeak']] = scaler.fit_transform(df[['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']])

                        
                return df
         

        # Function that maakes predictions and displays it on activation failed: "No access to GitHub Copilot found. You are currently logged in as mohamedhanfi."
        def make_predictions():
                
             # Cols with the order of the features trained on the model   
             reindexed_columns=['age','sex','chest pain type','resting bp s','cholesterol','fasting blood sugar','resting ecg','max heart rate','exercise angina','oldpeak','ST slope']
             
             # Saving the user inputs data into a dataframe
             df=save_user_inputs()
       
             # Scaling the user input data
             df=scaling_data(df)

             # Reindexing the user inputs dataframe to fit the order of the model order    
             df = df.reindex(columns=reindexed_columns)
             print(df)

             # Using the loaded model to Predict the price 
             dt_model=self.controller.load_dt_model()
             pred=dt_model.predict(df)
             pred = pred.reshape(1, -1)
             # Updating the prediction label with the predicted class
             self.prediction_label.config(text=f"Prediction: {(pred[0][0])}")     
             # Updating the prediction label with the predicted class
             prediction_value = pred[0][0]
             if prediction_value == 1:
                self.prediction_label.config(text="Result: Positive", fg="red")
             elif prediction_value == 0:
                self.prediction_label.config(text="Result: Negative",fg="green")



# Age
        age_entry_frame = tk.Frame(self, bg='#041618')
        age_entry_frame.pack(pady=5)
        age_entry_label = tk.Label(age_entry_frame, text="Age:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        age_entry_label.grid(row=0, column=0, padx=10, pady=0, sticky='w')
        self.age_entry = tk.Entry(age_entry_frame, bg='#FFFFC7', fg='#041618', font=("Arial", 12))  
        self.age_entry.grid(row=0, column=1, padx=10, pady=5)

# Sex
        Sex_frame = tk.Frame(self, bg='#041618')
        Sex_frame.pack(pady=5)
        Sex_label = tk.Label(Sex_frame, text="Sex:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        Sex_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.Sex = tk.StringVar(self) 
        self.Sex.set("Male")
        Sex_menu = tk.OptionMenu(Sex_frame, self.Sex, "Male", "Female")
        Sex_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        Sex_menu.grid(row=0, column=1, padx=10, pady=5)

# Chest Pain Type
        chest_pain_type_frame=tk.Frame(self, bg='#041618')
        chest_pain_type_frame.pack(pady=5)
        chest_pain_type_label=tk.Label(chest_pain_type_frame, text="Chest Pain Type:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        chest_pain_type_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.chest_pain_type=tk.StringVar(self)
        self.chest_pain_type.set("Typical angina")
        chest_pain_type_menu=tk.OptionMenu(chest_pain_type_frame,self.chest_pain_type,"Typical angina","Atypical angina","Non-anginal pain"," Asymptomatic")
        chest_pain_type_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        chest_pain_type_menu.grid(row=0, column=1, padx=10, pady=5)

# Restingbps
        resting_bps_frame = tk.Frame(self, bg='#041618')
        resting_bps_frame.pack(pady=5)
        resting_bps_label = tk.Label(resting_bps_frame, text="Resting bps:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        resting_bps_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.resting_bps= tk.Entry(resting_bps_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.resting_bps.grid(row=0, column=1, padx=10, pady=5)

#cholesterol
        cholesterol_frame = tk.Frame(self, bg='#041618')
        cholesterol_frame.pack(pady=5)
        cholesterol_label = tk.Label(cholesterol_frame, text="Cholesterol:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 16))
        cholesterol_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.cholesterol = tk.Entry(cholesterol_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.cholesterol.grid(row=0, column=1, padx=10, pady=5)

#fastingblood sugar
        fasting_blood_sugar_frame = tk.Frame(self, bg='#041618')
        fasting_blood_sugar_frame.pack(pady=5)
        fasting_blood_sugar_label = tk.Label(fasting_blood_sugar_frame, text="Fast Blood Sugar:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        fasting_blood_sugar_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.fasting_blood_sugar = tk.StringVar(self)
        self.fasting_blood_sugar.set("Fasting blood sugar > 120 mg/dl")
        fasting_blood_sugar_menu = tk.OptionMenu(fasting_blood_sugar_frame, self.fasting_blood_sugar, "Fasting blood sugar > 120 mg/dl", "Fasting blood sugar < 120 mg/dl")
        fasting_blood_sugar_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        fasting_blood_sugar_menu.grid(row=0, column=1, padx=10, pady=5)
       
#Resting ecg 
        resting_ecg_frame = tk.Frame(self, bg='#041618')
        resting_ecg_frame.pack(pady=5)
        resting_ecg_label = tk.Label(resting_ecg_frame, text="Resting ecg:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        resting_ecg_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.resting_ecg = tk.StringVar(self)
        self.resting_ecg.set("Normal")
        resting_ecg_menu = tk.OptionMenu(resting_ecg_frame, self.resting_ecg,"Normal","Having ST-T wave abnormality","Showing probable or definite left ventricular hypertrophy by Estes criteria")
        resting_ecg_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        resting_ecg_menu.grid(row=0, column=1, padx=10, pady=5)
# Max Heart Rate

        max_heart_rate_frame = tk.Frame(self, bg='#041618')
        max_heart_rate_frame.pack(pady=5)
        max_heart_rate_label = tk.Label(max_heart_rate_frame, text="Max Heart Rate:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        max_heart_rate_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.max_heart_rate = tk.Entry(max_heart_rate_frame, bg='#FFFFC7', font=("Helvetica", 12))
        self.max_heart_rate.grid(row=0, column=1, padx=10, pady=5)

# Exercise Angine
        exercise_angine_frame = tk.Frame(self, bg='#041618')
        exercise_angine_frame.pack(pady=5)
        exercise_angina_label = tk.Label(exercise_angine_frame, text="Exercise Angine:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        exercise_angina_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.exercise_angina = tk.StringVar(self)
        self.exercise_angina.set("NO")
        exercise_angina_menu = tk.OptionMenu(exercise_angine_frame, self.exercise_angina, "YES", "NO")
        exercise_angina_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
      
        exercise_angina_menu.grid(row=0, column=1, padx=10, pady=5)
       

# Old Peaks
        oldpeak_frame = tk.Frame(self, bg='#041618')
        oldpeak_frame.pack(pady=5)
        oldpeak_label = tk.Label(oldpeak_frame, text="Old Peaks:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        oldpeak_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.oldpeak = tk.IntVar(self)
        self.oldpeak.set(0)
        oldpeak_spinbox = tk.Spinbox(oldpeak_frame, from_=-10, to=10, increment=1, textvariable=self.oldpeak, font=("Helvetica", 12))
        oldpeak_spinbox.grid(row=0, column=1, padx=10, pady=5)
       
# ST Slope
        ST_slope_frame = tk.Frame(self, bg='#041618')
        ST_slope_frame.pack(pady=5)
        ST_slope_label = tk.Label(ST_slope_frame, text="ST Slope:", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        ST_slope_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.ST_slope = tk.StringVar(self)
        self.ST_slope.set("Upsloping")
        ST_slope_menu = tk.OptionMenu(ST_slope_frame, self.ST_slope,"Upsloping","Flat","Downsloping")
        ST_slope_menu.config(bg='#FFFFC7', font=("Helvetica", 12))
        ST_slope_menu.grid(row=0, column=1, padx=10, pady=5)
        
 # Label to display the prediction result
        self.prediction_label = tk.Label(self, text="", bg='#041618', fg="#FFFFC7", font=("Times New Roman", 18))
        self.prediction_label.pack(pady=0)
        # Button to trigger the function
        process_button = tk.Button(self, text="Predict", command=make_predictions, bg='#FFFFC7', font=("Helvetica", 14))
        process_button.pack(pady=0)

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
    def load_dt_model(self):
       DtModel = joblib.load('./Decision Tree/dtmodel.pkl')
       return DtModel
if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
