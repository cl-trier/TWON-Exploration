import typing

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TopicClassifier:
    # https://huggingface.co/cardiffnlp/tweet-topic-21-multi
    # https://arxiv.org/abs/2209.09824

    def __init__(self, slug: str = 'cardiffnlp/tweet-topic-21-multi'):
        self.tokenizer = AutoTokenizer.from_pretrained(slug)
        self.model = AutoModelForSequenceClassification.from_pretrained(slug)

        self.normalize_fn = torch.nn.Sigmoid()

    def __call__(self, batch: pd.Series, theta: float) -> pd.Series:
        return pd.Series(
            index=batch.index,
            data= self.extract_label(self.model_forward(batch.values()), theta)
        )

    def model_forward(self, batch: typing.List[str]) -> torch.tensor:
        return self.model(**self.tokenizer(batch, padding=True, return_tensors="pt")).logits

    def extract_label(self, batch_logits: torch.tensor, theta: float) -> typing.Iterator[typing.Set[str]]:
        batch_norm_logits: torch.tensor = (self.normalize_fn(batch_logits) >= theta).int()
        batch_ids: torch.tensor = [preds.nonzero().squeeze().tolist() for preds in torch.unbind(batch_norm_logits)]

        for post_ids in batch_ids:

            if isinstance(post_ids, list):
                yield {self.model.config.id2label[i] for i in post_ids}

            else:
                yield {self.model.config.id2label[post_ids], }
