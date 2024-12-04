from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def plant_disease_detection(image_path):
    # Load pre-trained model
    model = load_model(r'C:\\Users\\grand\\Downloads\\plant_disease_detection-main\\model.h5')
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale the image
    
    # Predict the disease
    prediction = model.predict(img_array)
    # Assuming that model returns probabilities for each class
    predicted_class = np.argmax(prediction, axis=1)
    
    # Return the predicted class
    return predicted_class

# Example usage:
# plant_disease_detection('path_to_image.jpg')
