import tensorflow as tf

loaded_model = tf.saved_model.load('./distillbert_sentiment_model')
print(loaded_model.signatures["serving_default"].structured_input_signature)

