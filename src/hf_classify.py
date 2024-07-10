import typing

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm

class HFClassify:
    # https://huggingface.co/cardiffnlp/tweet-topic-21-multi
    # https://arxiv.org/abs/2209.09824

    def __init__(self, slug: str = 'cardiffnlp/tweet-topic-21-multi'):
        self.tokenizer = AutoTokenizer.from_pretrained(slug)
        self.model = AutoModelForSequenceClassification.from_pretrained(slug)

        self.normalize_fn = torch.nn.Sigmoid()

    def __call__(self, samples: pd.Series, theta: float, batch_size: int = 32) -> pd.Series:
        return pd.concat([
            pd.Series(
                index=batch.index,
                data=self.extract_label(self.model_forward(list(batch.array)), theta)
            )
            for _, batch in tqdm.tqdm(samples.groupby(np.arange(len(samples)) // batch_size))
        ])

    def model_forward(self, batch: typing.List[str]) -> torch.tensor:
        return self.model(**self.tokenizer.batch_encode_plus(batch, truncation=True, return_tensors="pt", max_length=512, padding='max_length')).logits

    def extract_label(self, batch_logits: torch.tensor, theta: float) -> typing.Iterator[typing.Set[str]]:
        batch_norm_logits: torch.tensor = (self.normalize_fn(batch_logits) >= theta).int()
        batch_ids: torch.tensor = [preds.nonzero().squeeze().tolist() for preds in torch.unbind(batch_norm_logits)]

        for post_ids in batch_ids:

            if isinstance(post_ids, list):
                yield {self.model.config.id2label[i] for i in post_ids}

            else:
                yield {self.model.config.id2label[post_ids], }
