import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import librosa as lb
import librosa.display

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# GLOBAL VARIABLES
# -----------------
H = 128 # height of the tensor
N_FEATURES = 1 # Channels of the tensor
AUG_FEATURES = 5
N_SLICES = 3

def setup_vars(labels): 
    """
    Calculate setup variables oh_dict and n_files

    Args:
        labels: list of labels / folders
        
    Returns:
        oh_dict: one hot vector mapping each label to its one-hot vector. 
        n_files: number of files to be read.  
    """

    indices = [i for i in range(len(labels))]
    depth = len(labels)
    oh = tf.one_hot(indices, depth)
    oh_dict = {label:np.array(oh[i,:]) for i, label in enumerate(labels)}


    n_files = 0
    for folder in labels:
        files = os.listdir(path+'/'+folder)
        for file in files:
            if file.split('.')[-1] == 'wav':
                n_files += 1
    return oh_dict, n_files

def read_sample(file):
    y, sr = lb.load(file)
    return y, sr

def augment_sample(sample, sr):
    """
    Data augmentation. To prevent overfitting
    Techniques used: pitch shist, time stretch, add noise

    Args:
        sample: usually output of lb.load()
        sr: sampling rate. output from lb.load()
        
    Returns:
        list with each augmented sample as an item of the list
    """
    augmented_samples = list()
    #. 1. Split the sample into smaller samples
    samples = np.array_split(sample, N_SLICES)
    
    for sample_ in samples:
        # 2. Add the original sample
        augmented_samples.append(sample_)
        # 3. Change pitch
        for n_steps in [4,6]:
            augmented_samples.append(lb.effects.pitch_shift(sample_, sr,n_steps))

        # 4. Time Stretch
        for rate in [0.5, 2.0]:
            augmented_samples.append(lb.effects.time_stretch(sample_, rate))

        # 5. White Noise
        wn = np.random.randn(len(sample_))
        augmented_samples.append(sample_ + 0.005*wn)
    
    return augmented_samples

def get_input_length(path, folders):
    """
    Get the input length from the first file
    """
    folder = folders[0]
    files = os.listdir(path+'/'+folder)
    input_length = 0
    for file in files:
        if file.split('.')[-1] == 'wav':
            sample, sr = read_sample(path+'/'+folder+'/'+file)
            input_length = len(sample) // N_SLICES
            break
        
    
    return input_length


def get_n_frames(sample, sr):
    mel = lb.feature.melspectrogram(sample, sr)
    return mel.shape[1]
    
    
def read_raw_data(path, folders, input_length, m):
    """
    Read the data from disk and save it in two numpy arrays. 
    Run only once, and then run create_dataset for computing 
    the different features.
    
    Args:
        path: path to where the folders with the audio data is
        folders: list of folders. in this case is the same as the labels
        input_length: length of the sequence
        m: number of rows in the input tensor
        
    Returns: 
        x_raw: samples plus augmented samples. 
         shape: (m, input_length)
        y_raw: y tensor of the samples and augmented samples
         shape: (m, number of classes) one-hot encoded
    
    """
    oh_dict, _ = setup_vars(folders)
    x_raw = np.zeros((m, input_length))
    y_raw = np.zeros((m, len(labels)))
   
    i = 0
    for folder in folders:
        print(f"Reading {folder}")
        files = os.listdir(path+'/'+folder)
        for file in files:
            
            # make sure the file is a .wav file
            if file.split('.')[-1] == 'wav':
                sample, sr = read_sample(path+'/'+folder+'/'+file)
                aug_samples = augment_sample(sample, sr)
                for sample_ in aug_samples: 
                    # padding or shorten sample
                    if len(sample_) > input_length:
                        sample_ = sample_[:input_length]
                    else:
                        sample_ = np.pad(sample_, (0, max(0, input_length - len(sample_))))
                    x_raw[i,:] = sample_
                    y_raw[i,:] = oh_dict[folder]
                    i += 1
    if args.verbose:
        print('Samples read: ', i)
    return x_raw, y_raw, sr
    
def create_dataset(x_raw, sr, m):
    """
    Args:
        x_raw: Tensor with shape (m, input_length)
        sr: The sr used across the samples. same for all
        m: number of rows in the input tensor 
    
    Returns: 
        X: Tensor(m, 128, n_frames, N_FEATURES). With the features
        already encoded
        
    """
    n_frames = get_n_frames(x_raw[0,:], sr)
    X = np.zeros((m, 128, n_frames, N_FEATURES))

    for i in range(m):
        sample = x_raw[i,:]
        mel = lb.feature.melspectrogram(sample, sr)
        spect = lb.power_to_db(mel, ref=1.0)
        norm_spect = normalize(spect)
        X[i,:,:,0] = norm_spect
      
    return X

def lenet(input_shape, n_classes):
    """
    Build the LeNet network. 
    2 conv layers, 2 FC layers, average pooling and dropout. 

    Args:
        input_shape: input shape of the data for the Input layer
        n_classes: output vector shape
        
    Returns:
        inputs and outputs to be used in the Model call (keras.models.Model())
    """

    inputs = keras.Input(shape=input_shape) 

    x = layers.Conv2D(8, (3,3), activation='relu')(inputs)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(16, (3,3), activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(84, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation = 'softmax')(x)

    return inputs, outputs

def cnn_64(input_shape, n_classes):
    """
    Build the CNN 64 network explained in the report. 
    4 conv layers, max pooling and dropout. 

    Args:
        input_shape: input shape of the data for the Input layer
        n_classes: output vector shape
        
    Returns:
        inputs and outputs to be used in the Model call (keras.models.Model())
    """

    n_filters = 64
    inputs = keras.Input(shape=input_shape) 
    x = layers.Conv2D(n_filters, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(n_filters, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(n_filters, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(n_filters, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    return inputs, outputs

def pad_mel(mel, input_shape):
    """
    Reshape mel so that it matches the input_shape of the model

    Args:
        mel: mel spectogram. 2D array
        input_shape: input shape of the data for the Input layer
        
    Returns:
        Reshaped / padded spectogram
    """
    input_length = input_shape[2]
    if mel.shape[1] == input_length:
        return mel

    elif mel.shape[1] > input_length:
        mel = mel[:,:input_length]
    else:
        mel = np.pad(mel, (0, max(0, mel- len(mel))))
    return mel

def train(data_path, model_name, epochs=10, lr=0.01):
    """
    Train model_mame on the dataset on data_path.
    First we obtain the features, then train the model.
    Saves the model if -s is True
    Args:
        data_path: where is the folder containing the data
        model_name: lenet or cnn64
        epochs: number of epochs
        lr: learning rate
    Returns:
        Nothing
    """

    # Data
    _, n_files = setup_vars(labels)
    m = n_files * N_SLICES * (AUG_FEATURES + 1)
    input_length = get_input_length(path, labels)
    x_raw, y_raw, sr = read_raw_data(data_path, labels, input_length, m)
    X = create_dataset(x_raw, sr, m)
    Y = np.copy(y_raw)
    assert X.shape[0] == Y.shape[0], "X, Y shapes don't match"
    assert Y.shape[1] == len(labels), "Y shape doesn't match the shape of the labels"
    
    # train, validation, test split
    x_train, x_test_valid, y_train, y_test_valid = train_test_split(X,Y, train_size=0.8)
    x_val, x_test, y_val, y_test = train_test_split(x_test_valid,y_test_valid, test_size=0.5)

    if args.verbose:
        print(f'Shapes.\n\tx_raw: {x_raw.shape}, y_raw{y_raw.shape}')
        print(f'\tX: {X.shape}, Y: {Y.shape}')
        print(f'\tTrain: x {x_train.shape} y {y_train.shape}')
        print(f'\tValid: x {x_val.shape} y {y_val.shape}')
        print(f'\tTest: x {x_test.shape} y {y_test.shape}')

    # Model
    input_shape = (X.shape[1], X.shape[2], X.shape[3])
    if model_name.startswith('lenet'):
        inputs, outputs = lenet(input_shape, len(labels))
    elif model_name.startswith('cnn64'):
        inputs, outputs = lenet(input_shape, len(labels))
    else:
        raise Exception("Training currently supporting only lenet or cnn64 models")

    model = models.Model(inputs=inputs, outputs=outputs)
    optimizer = optimizers.Adam(learning_rate = lr)
    model.compile(optimizer, loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])
    if args.verbose:
        model.summary()

    history = model.fit(x=x_train, 
                        y=y_train,
                        validation_data=(x_val, y_val),
                        epochs=epochs,
                        shuffle=True)
    print("Evaluating on Test Set")
    model.evaluate(x=x_test, y=y_test)
    if args.save:
        model.save('../models/model1')

    
def predict(path_file, model_name='lenet2'):
    """
    Predict the music genre of 'path_file'
    from 'model'.

    Args:
        path_file: path to the file
        model_name: Optional. The mode to use. If no model is provided the default model is used.
        
    Returns:
        Doesn't return anything
    """

    if os.path.isfile(path_file):
        if (path_file.split('.')[-1] == 'wav'):
            sample, sr = read_sample(path_file)
            mel = lb.feature.melspectrogram(sample, sr)
            spect = lb.power_to_db(mel, ref=1.0)
            norm_spect = normalize(spect)

            # Load model
            model = models.load_model('../models/'+str(model_name))
            input_shape = model.layers[0].output_shape[0]

            x = pad_mel(norm_spect, input_shape)
            x = np.reshape(x, (1, input_shape[1], input_shape[2], input_shape[3]))

            if args.verbose:
                print(model.summary())
                print(f"input_shape: {input_shape} mel shape: {norm_spect.shape}")
                print(f"mel shape after padding {x.shape}")

            y_pred = model.predict(x)
            label_pred = np.argmax(y_pred)
            print(f"----\nThe predicted genre is {labels[label_pred]} wih probability {np.max(y_pred)}")
        else:
            raise Exception("Only wav files are allowed")
    else:
        raise FileNotFoundError()
    


if __name__ == "__main__":
    

    # Parsing arguents
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="train or predict")
    parser.add_argument("data",  help="path to the data. e.g. ../data/genres. If train, for training on that data, if predict, to get the labels")
    parser.add_argument("-f",  "--file", help="file to predict")
    parser.add_argument("-m", "--model",  help="trained model from /models/ to use for prediction. otherwise using default")
    parser.add_argument("-v", "--verbose",  help="verbose output", action="store_true")
    parser.add_argument("-s", "--save",  help="save model after training", action="store_true")

    args = parser.parse_args()


    # Setup
    path = args.data
    labels = [f.name for f in os.scandir(path) if f.is_dir()]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    epochs = 10 

    # Train or Predict
    if args.model:
        model = args.model
    else:
        model = 'lenet2'

    if args.train == "train":
        print(f"Training\n---------\nData: {args.data}")
        print(f'Model: {model}')
        train(args.data, model, epochs=epochs)

    elif args.train == "predict":
        file = args.file
        print(f"Predicting\n---------\nData: {args.data} \nFile: {file}")
        print(f'Model: {model}')
        predict(file, args.model)


    else:
        raise Exception("Incorrect keyword in the argument. train or predict are allowed")


    
