from __future__ import print_function

# Networks
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.preprocessing.image import ImageDataGenerator

# Layers
from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

# Utils
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime

# Files
import utils

# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

def createEmptyVGG16Base(input_shape):
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),



    #Flatten(),
    #Dense(4096, activation='relu'),
    #Dense(4096, activation='relu'),
    #Dense(1000, activation='softmax')
    ])
    return model


def createEmptyVGG19Base(input_shape):
    model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),


    
    #Flatten(),
    #Dense(4096, activation='relu'),
    #Dense(4096, activation='relu'),
    #Dense(1000, activation='softmax')
    ])
    return model




# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--image', type=str, default="/data/home/jonas/Images/test/Tallstam/IMG_0702.jpg", help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="Images", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=float, default=20.0, help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')
parser.add_argument('--model', type=str, default="VGG19", help='Your pre-trained classification model of choice')
parser.add_argument('--images_path', type=str, default="/home/jonas/Images/", help="Full path to images dataset")
parser.add_argument('--include_top', type=str2bool, default=False, help="Include top layers")
parser.add_argument('--weights', type=str, default="imagenet", help="which pretrained weights (imagenet, none)")#imagenet
parser.add_argument('--train_from_scratch', type=str2bool, default=False, help="Train model from scratch")
parser.add_argument('--base_layers_trainable', type=str2bool, default=False, help="Base layers trainable")

args = parser.parse_args()


# Global settings
BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
TRAIN_DIR = args.images_path + "/train/"
VAL_DIR = args.images_path + "/val/"
INCLUDE_TOP =  args.include_top
WEIGHTS = args.weights



preprocessing_function = None
base_model = None


# Prepare the model
if args.model == "VGG16":
    from keras.applications.vgg16 import preprocess_input
    preprocessing_function = preprocess_input
    if args.train_from_scratch:
        base_model = createEmptyVGG16Base(input_shape=(HEIGHT, WIDTH, 3))
    else:
        base_model = VGG16(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "VGG19":
    from keras.applications.vgg19 import preprocess_input
    preprocessing_function = preprocess_input
    if args.train_from_scratch:
        base_model = createEmptyVGG19Base(input_shape=(HEIGHT, WIDTH, 3))
    else:
        base_model = VGG19(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "ResNet50":
    from keras.applications.resnet50 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = ResNet50(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionV3":
    from keras.applications.inception_v3 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionV3(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "Xception":
    from keras.applications.xception import preprocess_input
    preprocessing_function = preprocess_input
    base_model = Xception(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "InceptionResNetV2":
    from keras.applications.inceptionresnetv2 import preprocess_input
    preprocessing_function = preprocess_input
    base_model = InceptionResNetV2(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "MobileNet":
    from keras.applications.mobilenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = MobileNet(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet121":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet121(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet169":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet169(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "DenseNet201":
    from keras.applications.densenet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = DenseNet201(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetLarge":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetLarge(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
elif args.model == "NASNetMobile":
    from keras.applications.nasnet import preprocess_input
    preprocessing_function = preprocess_input
    base_model = NASNetMobile(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
else:
    ValueError("The model you requested is not supported in Keras")

    
if args.mode == "train":
    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Resize Height -->", args.resize_height)
    print("Resize Width -->", args.resize_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tRotation -->", args.rotation)
    print("\tZooming -->", args.zoom)
    print("\tShear -->", args.shear)
    print("\images_path -->", args.images_path)
    print("")

    # Create directories if needed
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")

    # Prepare data generators
    train_datagen =  ImageDataGenerator(
      rescale = 1./255,
      preprocessing_function=preprocessing_function,
      rotation_range=args.rotation,
      shear_range=args.shear,
      zoom_range=args.zoom,
      horizontal_flip=args.h_flip,
      vertical_flip=args.v_flip
    )

    val_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocessing_function)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(HEIGHT, WIDTH), 
        batch_size=BATCH_SIZE
        ,class_mode='categorical'
        )

    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR, 
        target_size=(HEIGHT, WIDTH), 
        batch_size=BATCH_SIZE,
        )


    # Save the list of classes for prediction mode later
    class_list = utils.get_subfolders(TRAIN_DIR)
    utils.save_class_list(class_list, model_name=args.model, dataset_name=args.dataset)


    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list), base_layers_trainable=args.base_layers_trainable)

    if args.continue_training:
        finetune_model.load_weights("./checkpoints/" + args.model + "_model_weights.h5")

    adam = Adam(lr=0.00001)
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    num_train_images = utils.get_num_files(TRAIN_DIR)
    num_val_images = utils.get_num_files(VAL_DIR)

    def lr_decay(epoch):
        if epoch%20 == 0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr/2)
            print("LR changed to {}".format(lr/2))
        return K.get_value(model.optimizer.lr)

    learning_rate_schedule = LearningRateScheduler(lr_decay)

    filepath="./checkpoints/" + args.model + "_model_weights.h5"
    
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    callbacks_list = [checkpoint]


    history = finetune_model.fit_generator(train_generator, epochs=args.num_epochs, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE, 
        validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)


    plot_training(history)

elif args.mode == "predict":

    if args.image is None:
        ValueError("You must pass an image path when using prediction mode.")

    # Create directories if needed
    if not os.path.isdir("%s"%("Predictions")):
        os.makedirs("%s"%("Predictions"))
        
    modell = load_model("./checkpoints/" + args.model + "_model_weights.h5")

    # Read in your image
    image = cv2.imread(args.image,-1)
    save_image = image
    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
    #image = image / 255.0
    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))


    result = modell.predict(image)
    print(result)


    class_list_file = "./checkpoints/" + args.model + "_" + args.dataset + "_class_list.txt"

    class_list = utils.load_class_list(class_list_file)
    
    #finetune_model = utils.build_finetune_model(base_model, len(class_list))
    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS, num_classes=len(class_list), base_layers_trainable = args.base_layers_trainable)





    finetune_model.load_weights("./checkpoints/" + args.model + "_model_weights.h5")

    # Run the classifier and print results
    st = time.time()

    out = finetune_model.predict(image)

    confidence = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]

    run_time = time.time()-st

    print("Predicted class = ", class_name)
    print("Confidence = ", confidence)
    print("Run time = ", run_time)
    cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)
