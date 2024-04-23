import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

#All pages
class Page(tk.Frame):
    def __init__(self, parent, controller, title):
        tk.Frame.__init__(self, parent, bg='#041618')
        self.controller = controller
        
        label = tk.Label(self, text=title, bg='#041618', fg="#FFFFC7", font=("Helvetica", 32), width=20, height=2)
        label.pack(pady=10, padx=10)

        if not isinstance(self, StartPage):
            button_back = tk.Button(self, text="Back", width=10, height=1, bg='#FFFFC7', command=lambda: controller.show_frame(StartPage))
            button_back.pack(pady=10, padx=10)

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


#SVM page
class PageTwo(Page):
    def __init__(self, parent, controller):
        Page.__init__(self, parent, controller, "SVM")

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
        self.maxsize(width=1000,height=500)

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

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
