import json
import logging
import typing

import pandas as pd
import pydantic
import requests
import tqdm


class PromptClassify(pydantic.BaseModel):
    name: str
    template: str
    classes: typing.Dict

    def __call__(self, samples: pd.Series, model: str, options: dict):
        predictions: typing.Dict[typing.Hashable, str | None] = {}

        for index, value in tqdm.tqdm(
                samples.items(),
                total=len(samples),
                desc=f"classifying {self.name}"
        ):
            try:
                predictions[index] = self.classes.get(
                    requests.post(
                        'https://inf.cl.uni-trier.de/',
                        json={
                            'model': model,
                            'system': self.template,
                            'prompt': self.template.format(text=value),
                            'options': options
                            }).json()['response'].strip(),
                    None
                )

            except Exception as _e:
                logging.warning(_e)
                predictions[index] = None

        return pd.Series(predictions, name=self.name)

    @classmethod
    def from_json(cls, path: str) -> "PromptClassify":
        with open(path, encoding='utf-8') as fp:    # I added encoding='utf-8' to fix the UnicodeDecodeError
            return cls(**json.load(fp))
