import tensorflow as tf
from cls.Yamnet import Yamnet

def load_audio(file_path, target_sr=16000):
    audio = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    if sr != target_sr:
        audio = tf.signal.resample(audio, int(len(audio) * target_sr / sr), len(audio))
    return audio.numpy(), sr.numpy()

audio, sr = load_audio('/Users/kennyromanov/Projects/open-spqd/tmp/output_1.wav')

yamnet = Yamnet()

result = yamnet.analyze(audio)
result = yamnet.index(result)

print(result)
