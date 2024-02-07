from dataclasses import dataclass
from typing import Optional

from data_config import data_config


@dataclass
class NERWord:
    value: str
    _labels: Optional[dict] = None

    def add_label(self, label: str):
        if self._labels is None:
            self._labels = {}
        self._labels[data_config.ner_labels[label]["type"]] = label

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def get_label_type(label):
        if label in data_config.coarse_ner_labels:
            return "coarse"
        if label in data_config.fine_ner_labels:
            return "fine"

        raise ValueError(f"Unknown label {label}")


def label_entity(entity: list[NERWord], entity_type: str) -> list[NERWord]:
    """
    Adds NER labels to each word of an entity.

    """
    for ind, word in enumerate(entity):
        if f"B-{entity_type}" not in data_config.ner_labels:
            label = "O"
        elif ind == 0:
            label = f"B-{entity_type}"
        else:
            label = f"I-{entity_type}"

        word.add_label(label)

    return entity


def ner_words_to_entities(words: list[NERWord]) -> dict:
    """
    Converts a list of NERWords to a list of entities.
    """
    entities = []
    open_entities = []
    for word in words:
        indices_to_close = []
        # check if open entities are continued in the next word
        for ind, entity in enumerate(open_entities):
            if f"I-{entity['name']}" in [word.labels[key] for key in word.labels]:
                entity["text"].append(word.value)
            else:
                indices_to_close.append(ind)
                entities.append(open_entities[ind])
        # close entities
        for ind in indices_to_close:
            open_entities.pop(ind)
        for _, label in word.labels.items():
            if label.startswith("B-"):
                entity = {"name": label[2:], "text": [word.value]}
                open_entities.append(entity)
    entities.extend(open_entities)
    return entities
