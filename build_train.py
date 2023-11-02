# function to create test, train and valid batches
import os
import argparse
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from keras.models import save_model
from utils import save_fig

PATH = os.getcwd()
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 50
TARGET_SIZE = (48,48)
NUM_CLASSES = 7
IMG_SIZE = 48

def save_modell(model,model_name):
    os.chdir(PATH)
    MODELS_PATH = PATH +'/models'
    if os.path.isdir(MODELS_PATH) is False:
        os.mkdir('models')
        MODELS_PATH = PATH + '/models'
    path = os.path.join(MODELS_PATH, model_name + ".h5")
    save_model(model, path)
    print('Model saved.')

def plot_training(history,name):
    hist = pd.DataFrame()
    hist["Train Loss"]=history.history['loss']
    hist["Validation Loss"]=history.history['val_loss']
    hist["Train Accuracy"]=history.history['accuracy']
    hist["Validation Accuracy"]=history.history['val_accuracy']

    fig, axarr=plt.subplots(nrows=2, ncols=1 ,figsize=(8,6))
    axarr[0].set_title("History of Loss in Train and Validation Datasets")
    hist[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])
    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")
    hist[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1])
    save_fig(name)
    plt.show()


def create_batches(path):
    train_path = path + '/train'
    test_path = path + '/test'
    valid_path = path + '/valid'

    #  Create a data augmentor
    data_augmentor = ImageDataGenerator(
                                samplewise_center=True,
                                rescale=1./255,
                                shear_range=0.2,
                                zoom_range = 0.2,
                                samplewise_std_normalization=True,
                                validation_split=0.2)


    train_batches = data_augmentor.flow_from_directory(
                                                directory=train_path,
                                                target_size=TARGET_SIZE,
                                                batch_size=BATCH_SIZE)


    valid_batches = data_augmentor.flow_from_directory(
                                                        directory=valid_path,
                                                        target_size=TARGET_SIZE,
                                                        batch_size=BATCH_SIZE)

    return (train_batches,valid_batches)


# define model architecture
def define_model(input_shape=(48, 48, 3), classes=7):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='valid',input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='valid'))

    model.add(Conv2D(64, (3, 3), activation='linear',padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='linear',padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))

    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(classes, activation='softmax'))

    return model

# compile and run the model
def compile_run_model(epoches):
    classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    # Training model from scratch

    model = define_model(input_shape=(48,48,3), classes=len(classes))
    model.summary()

    model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_batches,
                        epochs= epoches,
                        batch_size=BATCH_SIZE,
                        validation_data=valid_batches,verbose=2,
                        callbacks =[ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=1e-6),
                              tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,mode='min')])
    save_modell(model,'basic')
    plot_training(history,'basic_train')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Model")
    parser.add_argument("--dataset_path", required=True, help="Dataset Path")
    args = parser.parse_args()
    train_batches,valid_batches = create_batches(args.dataset_path)
    compile_run_model(EPOCHS)