"""
Created on 2 nov. 2023

@author: jcolley
"""

import time
import pprint
from datetime import datetime

from tensorflow import keras
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
import nutrig.eval.cnn_ds_template as dst
import numpy as np
import matplotlib.pyplot as plt

G_datadir = "/home/jcolley/projet/grand_wk/data/npy/dataset_tplate_1.0/"
G_model_tflite = G_datadir + "flt_2l_240920_150_ref.tflite"
G_model_tflite_i8 = G_datadir + "flt_2l_240920_150_ref_i8.tflite"


def tflite_inference_fp16_REF(input_data, f_tf="converted_model.tflite"):
    """
    input_data shape (1024,3) only
    """
    print(input_data.dtype)
    print(input_data.shape)
    # Load the TFLite model and allocate tensors.
    interpreter_fp16 = tf.lite.Interpreter(model_path=f_tf)
    interpreter_fp16.allocate_tensors()
    output_index = interpreter_fp16.get_output_details()[0]["index"]
    input_index = interpreter_fp16.get_input_details()[0]["index"]
    print(input_data.shape)
    print(input_data[0].shape)
    input_fmt = np.expand_dims(input_data[0], axis=0).astype(np.float32)
    print(input_fmt.shape)
    interpreter_fp16.set_tensor(input_index, input_fmt)
    interpreter_fp16.invoke()
    output = interpreter_fp16.get_tensor(output_index)
    print(output.shape)
    print(output.dtype)
    print(output)


def tflite_inference_fp16(input_data, f_tf="converted_model.tflite"):
    print(input_data.dtype)
    print(input_data.shape)
    # Load the TFLite model and allocate tensors.
    interpreter_fp16 = tf.lite.Interpreter(model_path=f_tf)
    interpreter_fp16.allocate_tensors()
    output_index = interpreter_fp16.get_output_details()[0]["index"]
    input_index = interpreter_fp16.get_input_details()[0]["index"]
    input_fmt = np.expand_dims(input_data[0], axis=0).astype(np.float32)
    interpreter_fp16.set_tensor(input_index, input_fmt)
    interpreter_fp16.invoke()
    output = interpreter_fp16.get_tensor(output_index)
    return output


def tflite_inference(input_data, pn_tf="converted_model.tflite"):
    """
    https://ai.google.dev/edge/litert/inference?hl=fr#run-python

    :param input_data:
    :param f_tf:
    """
    print(input_data.dtype)
    print(input_data.shape)
    interpreter = Interpreter(pn_tf)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    d_input = {}
    d_input["shape"] = (10000, 1024, 3)
    input_details = interpreter.get_input_details()
    pprint.pprint(input_details)
    output_details = interpreter.get_output_details()
    nb_trace = input_data.shape[0]
    output_data = np.empty(nb_trace, dtype=np.float32)
    t_wall = datetime.now()
    t_cpu = time.process_time()
    for idx in range(nb_trace):
        input_fmt = np.expand_dims(input_data[idx], axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_fmt)
        interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        proba = interpreter.get_tensor(output_details[0]["index"])
        # print(proba.shape, proba)
        output_data[idx] = np.squeeze(proba)
    duration_cpu = time.process_time() - t_cpu
    duration_wall = datetime.now() - t_wall
    print(f"CPU time= {duration_cpu} s")
    print(f"Wall time= {duration_wall} s")
    print(output_data.shape)
    return output_data


def test_no_quant():
    data_tpl = dst.get_data_template(dst.f_test_nok)
    output_data = tflite_inference(data_tpl.astype(np.float32), G_model_tflite)
    print(output_data[:20])
    return output_data


def test_quant_i8():
    data_tpl = dst.get_data_template(dst.f_test_nok)
    output_data = tflite_inference(data_tpl.astype(np.float32), G_model_tflite_i8)
    print(output_data[:20])
    return output_data


def compare_quant():
    out_no_quant = test_no_quant()
    out_quant = test_quant_i8()
    diff = out_no_quant - out_quant
    plt.figure()
    plt.title(f"Histo difference, std={diff.std():.4f} ")
    plt.hist(diff)
    plt.grid()


if __name__ == "__main__":
    # test_no_quant()
    # test_quant_i8()
    compare_quant()
    #
    plt.show()
