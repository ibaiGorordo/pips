import onnxruntime
import numpy as np

path = "pips_simp.onnx"
session = onnxruntime.InferenceSession(path,
                                            providers=['CUDAExecutionProvider',
                                                       'CPUExecutionProvider'])

model_inputs = session.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]

model_outputs = session.get_outputs()
output_names = [model_outputs[i].name for i in range(len(model_outputs))]


xy = np.random.rand(1, 2, 2).astype(np.float32)
rgbs = np.random.rand(1, 8, 3, 480, 640).astype(np.float32)

outputs = session.run(output_names, {input_names[0]: xy,
                                      input_names[1]: rgbs})

[print(output.shape) for output in outputs]