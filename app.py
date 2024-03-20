#Importing all the required libraries
import sys
import torch
import pyttsx3 as tts
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget

'''#Setting up the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)'''
device = 'cpu'

#Defining the model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

#Loading the Trained model
model = ConvolutionalNeuralNetwork()
model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))
model.to(device)

#Defining the classes
classes = [
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Zero"
]

#Preprocessing the image
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#Initializing the text-to-speech engine
engine = tts.init()
engine.setProperty('rate', 120)

#Defining the prediction function
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = classes[output.argmax(1).item()]
    
    return prediction

#Defining the speech function
def speak_prediction(prediction):
    engine.say(prediction)
    engine.runAndWait()

#Defining the main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digit Recognition TTS App")
        self.setGeometry(100, 100, 400, 300)

        #Create layouts
        main_layout = QVBoxLayout()
        image_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        #Create image label
        self.image_label = QLabel()
        image_layout.addWidget(self.image_label)

        #Create an open file button
        open_file_button = QPushButton("Open Image")
        open_file_button.clicked.connect(self.open_image)
        button_layout.addWidget(open_file_button)

        #Add layouts to the main layout
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)

        #Create a central widget and set the main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    #Defining the open image function
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            prediction = predict_image(file_path)
            speak_prediction(prediction)
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)

#Running the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())