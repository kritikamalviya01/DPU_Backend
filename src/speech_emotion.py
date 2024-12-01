import time
import os
import numpy as np
import pyaudio
import wave
import librosa
from scipy.stats import zscore
import logging
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import LSTM

# Set up logging
logging.basicConfig(level=logging.INFO)

class speechEmotionRecognition:
    def __init__(self, subdir_model=None):
        # Load prediction model
        if subdir_model is not None:
            self._model = self.build_model()
            self._model.load_weights(subdir_model)

        # Emotion encoding
        self._emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    def voice_recording(self, filename, duration=5, sample_rate=16000, chunk=1024, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)
        frames = []
        
        logging.info('* Start Recording *')
        stream.start_stream()
        start_time = time.time()

        while (time.time() - start_time) < duration:
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info('* End Recording *')

        wf = wave.open(filename, 'w')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    def mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
        mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
        mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        return np.asarray(mel_spect)

    def frame(self, y, win_step=64, win_size=128):
    # Check if the input is too short and pad if necessary
        if y.shape[2] < win_size:
            logging.warning("Input audio is shorter than the expected frame size. Padding with zeros.")
            padding = np.zeros((y.shape[0], y.shape[1], win_size))
            padding[:, :, :y.shape[2]] = y
            y = padding

        nb_frames = 1 + int((y.shape[2] - win_size) / win_step)
        frames = np.zeros((y.shape[0], nb_frames, y.shape[1], win_size)).astype(np.float16)

        for t in range(nb_frames):
            start = t * win_step
            end = start + win_size
            # Ensure not to exceed the length of y
            if end <= y.shape[2]:
                frames[:, t, :, :] = np.copy(y[:, :, start:end]).astype(np.float16)
            else:
                logging.warning("Frame exceeds audio length, stopping frame extraction.")
                break

        return frames


    def build_model(self):
        K.clear_session()
        input_y = Input(shape=(5, 128, 128, 1), name='Input_MELSPECT')

        y = TimeDistributed(Conv2D(64, kernel_size=(3, 3), padding='same'), name='Conv_1_MELSPECT')(input_y)
        y = TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT')(y)
        y = TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT')(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='MaxPool_1_MELSPECT')(y)
        y = TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT')(y)

        # Repeat for more layers as necessary
        for _ in range(3):  # You can adjust this for more layers
            y = TimeDistributed(Conv2D(128, kernel_size=(3, 3), padding='same'), name='Conv_{}_MELSPECT'.format(_+2))(y)
            y = TimeDistributed(BatchNormalization(), name='BatchNorm_{}_MELSPECT'.format(_+2))(y)
            y = TimeDistributed(Activation('elu'), name='Activ_{}_MELSPECT'.format(_+2))(y)
            y = TimeDistributed(MaxPooling2D(pool_size=(4, 4)), name='MaxPool_{}_MELSPECT'.format(_+2))(y)
            y = TimeDistributed(Dropout(0.2), name='Drop_{}_MELSPECT'.format(_+2))(y)

        y = TimeDistributed(Flatten(), name='Flat_MELSPECT')(y)
        y = LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1')(y)
        y = Dense(7, activation='softmax', name='FC')(y)

        model = Model(inputs=input_y, outputs=y)
        return model

    def predict_emotion_from_file(self, filename, chunk_step=16000, chunk_size=49100, predict_proba=False, sample_rate=16000):
        try:
            y, sr = librosa.core.load(filename, sr=sample_rate, offset=0.5)
            chunks = self.frame(y.reshape(1, 1, -1), chunk_step, chunk_size)
            chunks = chunks.reshape(chunks.shape[1], chunks.shape[-1])

            y = np.asarray(list(map(zscore, chunks)))
            mel_spect = np.asarray(list(map(self.mel_spectrogram, y)))

            mel_spect_ts = self.frame(mel_spect)
            X = mel_spect_ts.reshape(mel_spect_ts.shape[0],
                                      mel_spect_ts.shape[1],
                                      mel_spect_ts.shape[2],
                                      mel_spect_ts.shape[3],
                                      1)

            if predict_proba:
                predict = self._model.predict(X)
            else:
                predict = np.argmax(self._model.predict(X), axis=1)
                predict = [self._emotion.get(emotion) for emotion in predict]

            K.clear_session()
            timestamp = np.concatenate([[chunk_size], np.ones((len(predict) - 1)) * chunk_step]).cumsum()
            timestamp = np.round(timestamp / sample_rate)

            return [predict, timestamp]

        except Exception as e:
            logging.error("Prediction error: %s", e)
            return None, None

    def prediction_to_csv(self, predictions, filename, mode='w'):
        with open(filename, mode) as f:
            if mode == 'w':
                f.write("EMOTIONS\n")
            for emotion in predictions:
                f.write(str(emotion) + '\n')

# Example of usage
# ser = SpeechEmotionRecognition(subdir_model='path_to_model_weights.h5')
# ser.voice_recording('output.wav', duration=5)
# predictions, timestamps = ser.predict_emotion_from_file('output.wav')
# ser.prediction_to_csv(predictions, 'predictions.csv')
