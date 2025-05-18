import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import DepthwiseConv2D

class EcoscanModel:
    def __init__(self, model_path="converted_keras (16)/keras_model.h5", labels_path="converted_keras (16)/labels.txt"):
        # Load the model
        print("Loading model...")
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'DepthwiseConv2D': DepthwiseConv2D},
            compile=False
        )
        
        # Load labels
        with open(labels_path, "r") as f:
            self.class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        image: numpy array of shape (height, width, 3) in BGR format
        """
        # Resize to model input size
        processed = cv2.resize(image, (224, 224))
        processed = np.asarray(processed, dtype=np.float32)
        processed = processed.reshape(1, 224, 224, 3)
        processed = (processed / 127.5) - 1
        return processed
    
    def predict(self, image):
        """
        Make a prediction on an image.
        Returns: dict with class name and confidence score
        """
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(processed_image, verbose=0)
        index = np.argmax(prediction)
        confidence_score = float(prediction[0][index] * 100)
        
        # Get all class predictions
        all_predictions = [
            {
                'class': class_name,
                'confidence': float(pred * 100)
            }
            for class_name, pred in zip(self.class_names, prediction[0])
        ]
        
        return {
            'class': self.class_names[index],
            'confidence': confidence_score,
            'all_predictions': all_predictions
        }

if __name__ == "__main__":
    # Example usage
    import cv2
    
    # Initialize model
    model = EcoscanModel()
    
    # Load and predict on a test image
    image_path = "test_image.jpg"  # Replace with your test image
    try:
        image = cv2.imread(image_path)
        if image is not None:
            result = model.predict(image)
            print(f"\nPrediction: {result['class']} ({result['confidence']:.1f}% confidence)")
            print("\nAll predictions:")
            for pred in result['all_predictions']:
                print(f"{pred['class']}: {pred['confidence']:.1f}%")
        else:
            print(f"Could not load image: {image_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}") 