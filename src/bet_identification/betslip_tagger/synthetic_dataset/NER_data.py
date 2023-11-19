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

    @property
    def labels(self):
        return self._labels


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
            if f"I-{entity['name']}" in word.labels:
                entity["text"].append(word.value)
            else:
                indices_to_close.append(ind)
                entities.append(open_entities[ind])
        # close entities
        for ind in indices_to_close:
            open_entities.pop(ind)
        for label in word.labels:
            if label.startswith("B-"):
                entity = {"name": label[2:], "text": [word.value]}
                open_entities.append(entity)
    entities.extend(open_entities)
    return entities
