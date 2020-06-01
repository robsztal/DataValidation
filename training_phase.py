#CNN model imports
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard
import time

# Use in terminal for library download
#-----pip install requirements.txt----

NAME = "MY_CNN_MODEL_64x2-{}".format(int(time.time()))
tb = TensorBoard(log_dir='logs/{}'.format(NAME))

#LOADING DATA FOR TRAINING
def load_dataset():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # reshape from RGB to single channel
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    #to represent category as numbers use hotencoding
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # return preloaded valuse
    return X_train, y_train, X_test, y_test

#PREPROCESSING OF DATA
def process_data(train, test):
    #train_data = train.astype('float32')
    #test_data = test.astype('float32')
	# normalize to range 0-1
    #train_data = train_data / 255.0
    #test_data = test_data / 255.0

    train_data = tf.keras.utils.normalize(train.astype('float32'), axis=1)
    test_data = tf.keras.utils.normalize(test.astype('float32'), axis=1)
    return train_data, test_data

#MODEL STRUCTURE
def my_model():

    filters = 64
    conv_activation = 'relu'
    dense_activation = 'relu'
    kernel_initializer = 'he_uniform'
    neurons = 100
    output_activation = 'softmax'
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = 'categorical_crossentropy'

    model = tf.keras.models.Sequential()

    #input layer
    model.add(Conv2D(filters=filters, kernel_size=(3,3), input_shape=(28, 28, 1), activation=conv_activation, kernel_initializer=kernel_initializer))

    #Max pooling takes max value of 2x2 pixels -> 1 pixel
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=filters, kernel_size=(3,3), activation=conv_activation, kernel_initializer=kernel_initializer))
    model.add(MaxPool2D(pool_size=(2,2)))

    #fully connected layers
    model.add(Flatten())
    model.add(Dense(units=neurons, activation=dense_activation, kernel_initializer=kernel_initializer))
    
    #output layer
    model.add(Dense(10, activation=output_activation))
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    print("Model Compiled")
    return model

#EVALUATION OF MODEL
def eval_model(X_data, y_data, n_folds=5):
    scores, histories = list(), list()
    
    #KFold divides all samples in k groups of samples (folds) of equal sizes
    kfold = KFold(n_folds, shuffle=True, random_state=1) 
    for Xi_train, Xi_test in kfold.split(X_data):
        model = my_model()

        #select rows to train in model
        X_train, y_train, X_test, y_test = X_data[Xi_train], y_data[Xi_train], X_data[Xi_test], y_data[Xi_test]
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        _ , acc = model.evaluate(X_test, y_test, verbose=0)
        print('>%.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(211)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(212)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

#SUMARIZING PERFORMANCE
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()

def evaluate_model(X_data, y_data, ep):
    model = my_model()
    model.fit(X_data, y_data, epochs=ep,  callbacks =[tb], validation_split=0.3)

#initialization of project
def test_model():
    #load data
    X_train, y_train, X_test, y_test = load_dataset()
    #process data
    X_train, X_test = process_data(X_train, X_test)
    #evaluate model
    evaluate_model(X_train, y_train, 10)
    #scores, histories = eval_model(X_train, X_test)
    #show performance
    #summarize_performance(scores)
    #summarize_diagnostics(histories)

test_model()
    
    
