import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.utils import CustomObjectScope

# Define custom metrics
def dice_coef(y_true, y_pred):
    return tf.reduce_mean((2. * y_true * y_pred) / (y_true + y_pred + tf.keras.backend.epsilon()))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Model paths
unet_model_path = '/content/Unetmodel.keras'
resunet_model_path = '/content/ResUnetmodel.keras'
unet3plus_model_path = '/content/unet3plusResnet50.keras'

# Load models
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    unet_model = tf.keras.models.load_model(unet_model_path)
    resunet_model = tf.keras.models.load_model(resunet_model_path)
    unet3plus_model = tf.keras.models.load_model(unet3plus_model_path)

def visualize_predictions(models, model_labels, test_x, test_y, num_samples=3):
    indices = random.sample(range(len(test_x)), num_samples)

    for i in indices:
        x, y = test_x[i], test_y[i]

        """ Extracting the name """
        name = x.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  ## [H, W, 3]
        #image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ## Convert BGR to RGB
        image = cv2.resize(image, (W, H))        ## [H, W, 3]
        x_input = image / 255.0                  ## Normalize
        x_input = np.expand_dims(x_input, axis=0)  ## [1, H, W, 3]

        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ## Convert BGR to RGB
        image2 = cv2.resize(image2, (W, H))        ## [H, W, 3]

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        """ Plotting """
        plt.figure(figsize=(18, 6))

        # Display the original image
        plt.subplot(1, len(models) + 2, 1)
        plt.title("Image")
        plt.imshow(image2)
        plt.axis('off')

        # Display the true mask
        plt.subplot(1, len(models) + 2, 2)
        plt.title("True Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        # Display predictions from each model
        for idx, (model, label) in enumerate(zip(models, model_labels), start=3):
            y_pred = model.predict(x_input, verbose=0)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred >= 0.5
            y_pred = y_pred.astype(np.int32)

            plt.subplot(1, len(models) + 2, idx)
            plt.title(f"{label} Prediction")
            plt.imshow(y_pred, cmap='gray')
            plt.axis('off')

        plt.show()

# Prepare models and labels
models = [unet_model, resunet_model, unet3plus_model]
model_labels = ["UNet", "ResUNet", "UNet3PlusResNet50"]

# Visualize predictions for all models
visualize_predictions(models, model_labels, test_x, test_y, num_samples=3)
