import numpy as np
import pandas as pd
from src.betslip_tagger.synthetic_dataset.NER_data import NERWord, label_entity


class BetMaker:
    def __init__(self, templates: dict, fillers: dict):
        self.templates = templates
        self.fillers = fillers

    def make_component(self, name: str, templates: pd.DataFrame):
        """
        Makes a single component of a bet in NER dataset format.
        """
        component_words = []
        ind = np.random.randint(0, len(templates))
        res = dict(templates.iloc[ind].copy(deep=True))
        res["name"] = name
        template = res.pop("template").split()
        for ind, word in enumerate(template):
            if word in self.fillers.keys():
                entity = self.fillers[word].fill()
                entity_words = [NERWord(value=word) for word in entity.split()]
                # add fine-grain NER labels
                entity_words = label_entity(entity_words, word)
            else:
                entity_words = [NERWord(value=word)]
            component_words.extend(entity_words)
        # add coarse-grain NER labels
        component_words = label_entity(component_words, name)
        res["words"] = component_words
        res["text"] = " ".join([ner_word.value for ner_word in component_words])

        return res

    def make_bet(self):
        bet = {}
        bet["event"] = self.make_component("event", self.templates["event"])
        # many valid market templates for a given event
        bet["market"] = self.make_component(
            "market", self.templates["market"][self.templates["market"]["event_id"] == bet["event"]["id"]]
        )
        # one valid side template for a given market
        side_id = bet["market"]["side_id"]
        bet["side"] = self.make_component("side", self.templates["side"][self.templates["side"]["id"] == side_id])
        bet["odds"] = self.make_component("odds", self.templates["odds"])

        return bet
