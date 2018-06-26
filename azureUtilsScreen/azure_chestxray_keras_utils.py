### Copyright (C) Microsoft Corporation.  

from keras.layers import Dense, Dropout
from keras.models import Model
from keras_contrib.applications.densenet import DenseNetImageNet121
import keras_contrib


def load_model(modelPath):
    model = build_model(DenseNetImageNet121)
    model.load_weights(modelPath)
    #print(model.summary)
    return model

def build_model(crt_densenet_function):
    """

    Returns: a model with specified weights

    """
    # define the model, use pre-trained weights for image_net
    base_model = crt_densenet_function(input_shape=(224, 224, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='avg')

    x = base_model.output
    dropout = Dropout(0.1)(x)
    predictions = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__=="__main__":
    model = load_model('/home/sthorfully/datadrive01/amlwbShare/XVisionML/XVisionWorkspace/XVisionCheXNet/model.hdf5')
#    model = build_model(DenseNetImageNet121)
#    print(model.summary())
#    model = build_model(keras_contrib.applications.densenet.DenseNetImageNet201)
#    print(model.summary())

