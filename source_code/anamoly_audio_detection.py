import numpy as np
import librosa
import sounddevice as sd
from keras.models import load_model
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


# Set the path to the trained model
model_path = r"E:\project\Sounds\glass_sound_detection.h5"

# Set the number of mel frequency bins in the spectrogram
n_mels = 128

# Set the number of time steps in each segment of the spectrogram
n_steps = 128

# Load the trained model
model = load_model(model_path)

# Define a function to convert an audio array to a mel-spectrogram
def array_to_melspec(audio):
    # Convert to mel-spectrogram
    spec = librosa.feature.melspectrogram(audio, sr=22050, n_mels=n_mels)
    # Resize the spectrogram to n_steps x n_mels
    spec = librosa.util.fix_length(spec, n_steps, axis=1)
    # Convert to decibel scale
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec

# Define a function to record audio from the microphone
def record(duration):
    # Set the sample rate and number of channels
    sr = 22050
    channels = 1
    # Record the audio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=channels)
    sd.wait()
    # Convert to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio

# Define a function to predict the class label of a mel-spectrogram
def predict_class(spec):
    spec = np.expand_dims(spec, axis=0)
    prediction = model.predict(spec)
    label_map = {1: 'glass_breaking', 0: 'other'}
    predicted_label = np.argmax(prediction)
    return label_map[predicted_label],prediction.tolist()

# Listen for glass breaking sound
while True:
    print('Listening for glass breaking sound...')
    audio = record(duration=2)  # Record 1 second of audio
    spec = array_to_melspec(audio)  # Convert audio to mel-spectrogram
    predicted_label,predicted_prob = predict_class(spec)
    print(predicted_prob)  # Make a prediction
    if predicted_label == 'glass_breaking' and int(predicted_prob[0][0]) >= 80:
        print('Glass breaking sound detected!')
        break
