import tensorflow as tf
import numpy as np
import os
from utils import convert_tflite_int8


input_shape = (124, 129, 1)
save_dir = os.path.join(os.getcwd(), "models")
model_dir = os.path.join(save_dir, "saved_model")
label_path = os.path.join(save_dir, "labels.txt")

convert_tflite_int8(model_dir)