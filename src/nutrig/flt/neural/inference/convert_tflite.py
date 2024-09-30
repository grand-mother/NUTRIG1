"""
Created on 30 sept. 2024

@author: jcolley
"""

import tensorflow as tf
import os.path
import keras

def convert_tflite(pn_keras, kind="no"):
    assert  os.path.exists(pn_keras)
    model = keras.models.load_model(pn_keras)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if kind == "no":
        new_pn = pn_keras.replace(".keras", ".tflite")
    elif kind == "int8":
        new_pn = pn_keras.replace(".keras", "_i8.tflite")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        raise
    tflite_model = converter.convert()
    with open(new_pn, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    convert_tflite(
        "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/flt_2l_240920_150_ref.keras"
    )
    convert_tflite(
        "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/flt_2l_240920_150_ref.keras",
        "int8"
    )