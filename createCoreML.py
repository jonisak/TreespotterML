import argparse
import coremltools
import keras
from keras.models import load_model



# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, default="/data/home/jonas/Documents/Transfer-Learning-Suite/checkpoints/", help='Path to h5 keras weight file')
parser.add_argument('--model', type=str, default="VGG19_model_weights.h5", help='name of h5 file')
parser.add_argument('--class_labels', type=str, default="VGG19_Images_class_list.txt", help='name of class list file')

args = parser.parse_args()

scale = 1.0


model = load_model(args.filepath + args.model)


coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names=['image'],
                                                    class_labels=args.filepath + "VGG19_Images_class_list.txt",
                                                    output_names=['probabilities'],
                                                    image_input_names='image',
                                                    predicted_feature_name='class',
                                                    image_scale=scale,
                                                    red_bias=0,
                                                    green_bias=0,
                                                blue_bias=0
                                                )


                      
                                               
coreml_model.short_description = "Treemodel v0.1"
coreml_model.save('Tree.mlmodel')

