import tensorflow as tf
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Custom object to handle older model format
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Register the custom objects
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

# Load the model
model = tf.keras.models.load_model("converted_keras (16)/keras_model.h5", compile=False, custom_objects=custom_objects)

# Load the labels
class_names = open("converted_keras (16)/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame")
        continue

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # Remove the index number from the label (e.g., "0 Animal" -> "Animal")
    class_name = class_name.split(' ', 1)[1].strip()
    print(f"Class: {class_name}", end=" ")
    print(f"Confidence Score: {confidence_score * 100:.1f}%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows() 