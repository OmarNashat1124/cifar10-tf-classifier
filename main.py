from utils import load_data, evaluate_model
from model import cnn_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    model = cnn_model()
    
    early_stop = EarlyStopping(
    monitor='val_accuracy',      
    patience=5,                 
    restore_best_weights=True   
)
    
    history = history = model.fit(
    x_train, y_train,
    epochs=50,                  
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    verbose=1
)


    
    evaluate_model(model, x_test, y_test)

    model.save('cifar10_model.h5')

if __name__ == "__main__":
    main()
