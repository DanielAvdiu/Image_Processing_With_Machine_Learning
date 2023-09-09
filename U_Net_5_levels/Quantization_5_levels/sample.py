from tensorflow.keras.models import load_model
import tensorflow as tf

try:
    h5_model = load_model("U_Net_5_levels/Quantization_5_levels/files_quantized_5_levels_02/model_quantized_5_levels_02.h5")
    print("H5 model loaded successfully.")
except (OSError, ValueError):
    print("Failed to load H5 model. The file might be corrupted.")

try:
    tflite_model = tf.lite.Interpreter(model_path="U_Net_5_levels/Quantization_5_levels/files_quantized_5_levels_02/model.tflite")
    tflite_model.allocate_tensors()
    print("TFLite model loaded successfully.")
except (ValueError, FileNotFoundError):
    print("Failed to load TFLite model. The file might be corrupted.")
