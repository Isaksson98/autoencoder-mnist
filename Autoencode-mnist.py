from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
import random
import numpy as np

training_set, testing_set = mnist.load_data()

x_train, y_train = training_set
x_test, y_test = testing_set

def preproccess(x_train, x_test):
   
   x_train_reshape = x_train.reshape(x_train.shape[0], 784)
   x_test_reshape = x_test.reshape(x_test.shape[0], 784)

   x_train_reshape = x_train_reshape/255
   x_test_reshape = x_test_reshape/255
   
   return x_train_reshape, x_test_reshape

def plot_conv_data(x_test, output):
    fig, ((ax1,ax2,ax3,ax4, ax5),(bx1,bx2,bx3,bx4, bx5)) = plt.subplots(2, 5)

    random_img = random.sample(range(output.shape[0]),5)

    for i, ax in enumerate([ax1,ax2,ax3,ax4, ax5]):
        ax.imshow(x_test[random_img[i]].reshape(28,28),cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for j, bx in enumerate([bx1,bx2,bx3,bx4,bx5]):
        bx.imshow(output[random_img[j]].reshape(28,28), cmap='gray')
        bx.grid(False)
        bx.set_xticks([])
        bx.set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_data(x_test, outputs):
    outputs.append(x_test)
    fig, axes = plt.subplots(len(outputs),5)
    
    random_img=random.sample(range(outputs[0].shape[0]),5)
    

    #autoencoded
    for row_num, rows in enumerate(axes):
        for col_num, ax in enumerate(rows):
            print(col_num)
            ax.imshow(outputs[row_num][random_img[col_num]].reshape(28,28), cmap='gray')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def create_auto_encoder(hidden_layers):
    model = Sequential()
    model.add(Dense(units=hidden_layers,input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    
    return model

x_train_reshaped, x_test_reshaped = preproccess(x_train,x_test)

x_train_noisy = x_train_reshaped + np.random.normal(0, 0.6, size=x_train_reshaped.shape)
x_test_noisy = x_test_reshaped + np.random.normal(0, 0.6, size=x_test_reshaped.shape)


x_train_noisy = np.clip(x_train_noisy, a_min=0, a_max=1)
x_test_noisy = np.clip(x_test_noisy, a_min=0, a_max=1)

plot_conv_data(x_train_noisy, x_train_noisy)

conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28,28,1)))
conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))

conv_autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same'))
conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))

conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))

conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
conv_autoencoder.fit(x_train_noisy.reshape(60000,28,28,1),x_train_noisy.reshape(60000,28,28,1), epochs=10)

conv_output=conv_autoencoder.predict(x_test_noisy.reshape(10000,28,28,1))

plot_conv_data(x_train_noisy, conv_output)

#Single autoencoder
model_2=create_auto_encoder(2)
model_4=create_auto_encoder(4)
model_8=create_auto_encoder(8)
model_16=create_auto_encoder(16)
model_32=create_auto_encoder(32)
model_64=create_auto_encoder(64)

models = [model_2, model_4,model_8, model_16, model_32, model_64]
#models = [model_16, model_32]
outputs = []


#for i, mod in enumerate(models):
    #mod.compile(optimizer='adam', loss='mean_squared_error')
    #mod.fit(x_train_noisy, x_train_noisy, epochs=10, verbose=2)
    #outputs.append(mod.predict(x_test_noisy))

#plot_data(x_test_noisy, outputs)

