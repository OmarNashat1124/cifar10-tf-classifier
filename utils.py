import tensorflow as tf 
from tensorflow.keras import datasets   
import numpy as np

def load_data():
    """Loading CIFAR10 Dataset"""
    (x_train , y_train), (x_test,y_test) = datasets.cifar10.load_data()
    

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)

def evaluate_model(model, x_test, y_test):
    """Evaluate model accuracy."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")