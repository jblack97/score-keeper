from dataclasses import dataclass
from typing import Optional


@dataclass
class NERWord:
    value: str
    _labels: Optional[dict] = None

    def add_label(self, label: str):
        if self._labels is None:
            self._labels = {}
        self._labels[label] = 1


def label_entity(entity: list[NERWord], entity_type: str) -> list[NERWord]:
    """
    Adds NER labels to each word of an entity.

    """
    for ind, word in enumerate(entity):
        if ind == 0:
            word.add_label(f"B-{entity_type}")
        else:
            word.add_label(f"I-{entity_type}")

    return entity
