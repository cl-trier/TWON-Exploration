import typing
import logging

import requests

import numpy as np
import pandas as pd

import tqdm


class SentenceEmbedder:
    def __call__(self, data: pd.Series, prefix: str = "") -> typing.Dict[str, np.ndarray]:
        embeds: typing.Dict[str, np.ndarray] = {}

        for index, value in tqdm.tqdm(data.items(), total=len(data)):

            try:
                embed = np.array(requests.post(
                    'https://inf.cl.uni-trier.de/embed/',
                    json={'prompt': prefix + value}
                ).json()["response"])

            except Exception as _e:
                logging.warning(_e)
                embed = None

            embeds[index] = embed

        return embeds
