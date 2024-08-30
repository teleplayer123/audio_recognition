import os
import tensorflow as tf
import numpy as np
from audio_processing import train_val_label_data, make_spec_ds, build_audio_recognition_model, plot_history, take_first
from utils import save_labels, save_tf_model


data_dir = os.path.join(os.getcwd(), "data", "mini_speech_commands")
train_ds, val_ds, label_names = train_val_label_data(data_dir)

def squeeze(audio, labels):
	audio = tf.squeeze(audio, axis=-1)
	return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)
tmp_ds = take_first(train_spectrogram_ds)[0]

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = tmp_ds.shape[1:]
print("Input Shape: {}".format(input_shape))
save_labels(label_names)

model = build_audio_recognition_model(input_shape, train_spectrogram_ds, label_names)

history = model.fit(
		train_spectrogram_ds,
		validation_data=val_spectrogram_ds,
		epochs=10,
		callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

save_tf_model(model)
# plot_history(history)

#extract weights and biases from model
weights_and_biases = {}
for i, layer in enumerate(model.layers):
    weights_biases = layer.get_weights()
    if len(weights_biases) < 1:
        continue
    weights = weights_biases[0]
    biases = weights_biases[1]
    weights_and_biases[f'w{i}'] = weights
    weights_and_biases[f'b{i}'] = biases

# Save the weights and biases to a file
fname = "model_weights_biases.npz"
path = os.path.join(os.getcwd(), "models", fname)
np.savez(path, **weights_and_biases)
