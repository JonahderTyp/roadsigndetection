import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt


# Modell laden
model = load_model('traffic_sign_classifier.h5')

# Pfad zu neuen Bildern
new_images_path = 'gtsrb-german-traffic-sign/Test'

# Funktion zum Vorbereiten und Klassifizieren eines einzelnen Bildes
def classify_image(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Schleife durch alle neuen Bilder im Verzeichnis
for img_name in os.listdir(new_images_path):
    img_path = os.path.join(new_images_path, img_name)
    predicted_class = classify_image(img_path, model)
    print(f'Image: {img_name} - Predicted Class: {predicted_class[0]}')

    # Klassenbezeichnungen (Beispiel)
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
    'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
    'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]


# Schleife durch alle neuen Bilder im Verzeichnis
for img_name in os.listdir(new_images_path):
    img_path = os.path.join(new_images_path, img_name)
    predicted_class, confidence, predictions = classify_image(img_path, model)
    
    if predicted_class is not None:
        predicted_label = class_names[predicted_class]
        print(f'Image: {img_name} - Predicted Class: {predicted_label} ({confidence*100:.2f}% confidence)')

        # Visualisierung des Bildes und der Vorhersage
        img = image.load_img(img_path, target_size=(64, 64))
        plt.imshow(img)
        plt.title(f'Predicted: {predicted_label}\nConfidence: {confidence*100:.2f}%')
        plt.axis('off')
        plt.show()
    else:
        print(f'Image: {img_name} - Could not be classified')