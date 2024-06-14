import typing

import numpy as np
import torch


class DistanceMetric:
    def __init__(self, fn: typing.Callable = torch.nn.PairwiseDistance()):
        self.fn = fn

    def __call__(self, groups: typing.Dict):

        results: typing.Dict[typing.Tuple[str, str], float] = {}

        for label_1, c_1 in groups.items():
            for label_2, c_2 in groups.items():

                if (
                        (label_1, label_2) in results.keys() or
                        (label_2, label_1) in results.keys()
                ):
                    continue

                res = sum([
                    sum(self.fn(
                        torch.tensor(np.array(v_1)),
                        torch.tensor(np.array(c_2.tolist()))
                    )) / len(c_2)
                    for v_1 in c_1
                ]) / len(c_1)

                results[(label_1, label_2)] = res.item()

        return results
