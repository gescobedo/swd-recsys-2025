from ast import literal_eval
from typing import List, Union, Optional

import numpy as np
from enum import StrEnum, auto
from dataclasses import dataclass
from sklearn.utils.class_weight import compute_class_weight


class FeatureType(StrEnum):
    categorical = auto()
    continuous = auto()
    continuous_seq = auto()


@dataclass
class FeatureDefinition:
    name: str
    type: FeatureType
    class_weights: Optional[Union[str, List[float]]] = "balanced"


class InteractionFeature:
    def __init__(
        self, feature: FeatureDefinition, raw_values: List[Union[int, float, str]]
    ):
        """
        Helper class for different user features to ease their handling

        :param feature: the definition of the feature
        :param raw_values: the raw feature values for the individual users, can either be numeric values (int/float)
                            or string representations of a list of values
        """

        self.feature = feature
        self.is_categorical_feature = self.feature.type == FeatureType.categorical.value
        self.is_sequential_feature = (
            self.feature.type == FeatureType.continuous_seq.value
        )
        self.is_continuous_feature = self.feature.type == FeatureType.continuous.value
        self.n_values = len(raw_values)

        self._raw_values = raw_values
        if self.is_sequential_feature:
            self.sequence_values = np.stack(
                [literal_eval(val) for val in self._raw_values]
            )

        if self.is_categorical_feature:
            self.unique_values = tuple(sorted(set(self._raw_values)))
            self.n_unique_values = len(set(self._raw_values))
            self.value_map = {lbl: i for i, lbl in enumerate(self.unique_values)}
            self.encoded_values = np.array(
                [self.value_map[lbl] for lbl in self._raw_values]
            )
            if isinstance(feature.class_weights, list):
                self.class_weights = np.array(feature.class_weights)
            else:
                self.class_weights = compute_class_weight(
                    class_weight=feature.class_weights,
                    classes=np.unique(self.encoded_values),
                    y=self.encoded_values,
                )
            self.value_indices_groups = {
                lbl: np.argwhere(self.encoded_values == self.value_map[lbl]).flatten()
                for lbl in self.unique_values
            }

    def get_values(self):
        # if self.is_continuous_feature:
        #    self._raw_values = (self._raw_values-self._raw_values.min())/(self._raw_values.max()-self._raw_values.min())
        if self.is_categorical_feature:
            return self.encoded_values
        else:
            return (
                self.sequence_values if self.is_sequential_feature else self._raw_values
            )

    def count(self):
        if self.is_categorical_feature:
            return {k: len(v) for k, v in self.value_indices_groups.items()}
        else:
            return self.n_values

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"InteractionFeature(name={self.feature.name}, type={self.feature.type}, counts={self.count()})"
