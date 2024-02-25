import typing
import numpy as np
import pandas as pd
import tensorflow as tf
import fwk
from typing import Any


def class_map_from(csv_filename: str) -> dict[Any, Any]:
    class_map_df = pd.read_csv(csv_filename)
    class_map = dict(zip(class_map_df['index'], class_map_df['display_name']))

    return class_map


class YamnetAnalyze:
    scores: typing.Any
    embeddings: typing.Any
    spectrogram: typing.Any


class Yamnet:
    def __init__(self) -> None:
        self.model_filename = fwk.path('res', 'YAMNet')
        self.class_map_filename = f'{self.model_filename}/assets/yamnet_class_map.csv'
        self.model = tf.saved_model.load(self.model_filename)
        self.class_map: typing.Any = class_map_from(self.class_map_filename)

    def analyze(self, np_bytes: bytes) -> YamnetAnalyze:
        scores, embeddings, spectrogram = self.model(np_bytes)

        result: YamnetAnalyze = YamnetAnalyze()
        result.scores = scores
        result.embeddings = embeddings
        result.spectrogram = spectrogram

        return result

    def index(self, analyze: YamnetAnalyze) -> typing.Any:
        # Average probability for each class by time
        average_scores = np.mean(analyze.scores, axis=0)

        # Making a dictionary
        average_scores_dict = {self.class_map[i]: score for i, score in enumerate(average_scores)}

        return average_scores_dict
