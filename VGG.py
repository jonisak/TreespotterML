from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint#, LearningRateScheduler
from keras.preprocessing import image
import matplotlib.pyplot as plt
import argparse
import utils

# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default="imagenet", help="which pretrained weights (imagenet, none)")#imagenet
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--include_top', type=str2bool, default=False, help="Include top layers")
parser.add_argument('--images_path', type=str, default="/home/jonas/Images/", help="Full path to images dataset")
parser.add_argument('--batch_size', type=int, default=50, help='Number of images in each batch')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train for')

args = parser.parse_args()
WEIGHTS = args.weights
INCLUDE_TOP =  args.include_top
WIDTH = args.resize_width
HEIGHT = args.resize_height
TRAIN_DIR = args.images_path + "/train/"
VAL_DIR = args.images_path + "/val/"
TEST_DIR = args.images_path + "/test/"
BATCH_SIZE = args.batch_size
num_train_images = utils.get_num_files(TRAIN_DIR)
num_val_images = utils.get_num_files(VAL_DIR)

def trainModel():
        base_model = VGG16(weights=WEIGHTS, include_top=INCLUDE_TOP, input_shape=(HEIGHT, WIDTH, 3))
        # Freeze the layers except the last 4 layers
        for layer in base_model.layers[:-4]:
                layer.trainable = False
                
        # Check the trainable status of the individual layers
        for layer in base_model.layers:
                print(layer, layer.trainable)
    
        # Create the model
        model = models.Sequential()
        # Add the vgg convolutional base model
        model.add(base_model)
        # Add new layers
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))
        # Show a summary of the model. Check the number of trainable parameters
        model.summary()
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
                )
        validation_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
                        
        validation_generator = validation_datagen.flow_from_directory(
                VAL_DIR,
                target_size=(HEIGHT, WIDTH),
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=False)


        #adam = Adam(lr=0.00001)


        # Compile the model
        model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=1e-4),
                metrics=['acc'])
        #model.compile(loss='categorical_crossentropy',
        #        optimizer=adam,
        #        metrics=['acc'])




        filepath="./checkpoints/" + "VGG16" + "_model_weights.h5"
        checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
        callbacks_list = [checkpoint]
        # Train the model
        history = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_images // BATCH_SIZE ,
        epochs=args.num_epochs,
        validation_data=validation_generator,
        validation_steps=num_val_images // BATCH_SIZE,
        verbose=1, callbacks=callbacks_list)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
 
        epochs = range(len(acc))
 
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
 
        plt.figure()
 
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

import numpy as np
def validateData(filepath):
        model = load_model(filepath)
        print(model.summary())


        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(HEIGHT, WIDTH),
        batch_size=1,
        class_mode='categorical')
 
        # Get the filenames from the generator
        fnames = test_generator.filenames
 
        # Get the ground truth from generator
        ground_truth = test_generator.classes
 
        # Get the label to class mapping from the generator
        label2index = test_generator.class_indices
 
        # Getting the mapping from class index to class label
        idx2label = dict((v,k) for k,v in label2index.items())
 
        # Get the predictions from the model using the generator
        predictions = model.predict_generator(test_generator, steps=test_generator.samples/test_generator.batch_size,verbose=1)
        predicted_classes = np.argmax(predictions,axis=1)
 
        errors = np.where(predicted_classes != ground_truth)[0]
        print("No of errors = {}/{}".format(len(errors),test_generator.samples))
 
        # Show the errors
        for i in range(len(errors)):
                pred_class = np.argmax(predictions[errors[i]])
                pred_label = idx2label[pred_class]
     
                title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
                        fnames[errors[i]].split('/')[0],
                        pred_label,
                        predictions[errors[i]][pred_class])
     
                original = image.load_img('{}/{}'.format(TEST_DIR,fnames[errors[i]]))
                plt.figure(figsize=[7,7])
                plt.axis('off')
                plt.title(title)
                plt.imshow(original)
                plt.show()
        print('test')

import coremltools
from keras.models import load_model

def createCoreML(filepath):
        scale = 1./255
        model = load_model(filepath)
        print(model.summary())
        #model = load_model(args.filepath + args.model)
        coreml_model = coremltools.converters.keras.convert(model,
                input_names=['image'],
                class_labels=['Bjorkstam', 'Granstam', 'Tallstam'], # "VGG16_Images_class_list.txt",
                output_names=['probabilities'],
                image_input_names='image',
                predicted_feature_name='class',
                image_scale=scale,
                red_bias=-1.0,
                green_bias=-1.0,
                blue_bias=-1.0,
                is_bgr=False
        )
        coreml_model.short_description = "Treemodel v0.1"
        coreml_model.save('Tree.mlmodel')


#trainModel()
createCoreML("./checkpoints/" + "VGG16" + "_model_weights.h5")
validateData("./checkpoints/" + "VGG16" + "_model_weights.h5")

