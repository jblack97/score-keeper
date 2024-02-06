import numpy as np
import pandas as pd
from .NER_data import NERWord, label_entity, ner_words_to_entities


class BetMaker:
    def __init__(self, templates: dict, fillers: dict):
        self.templates = templates
        self.fillers = fillers

    def template_to_ner(self, template: str, fill_values: dict):
        component_words = []
        for _, word in enumerate(template.split()):
            if word in self.fillers:
                if word in fill_values:
                    entity = self.fillers[word].override_fill(fill_values[word])
                else:
                    entity = self.fillers[word].fill()
                entity_words = [NERWord(value=word) for word in entity.split()]
                # add fine-grain NER labels
                entity_words = label_entity(entity_words, word)
            else:
                entity_words = [NERWord(value=word)]
            component_words.extend(entity_words)

        return component_words

    def make_component(self, name, templates: pd.DataFrame, fill_values: dict = None):
        if fill_values is None:
            fill_values = {}
        res = dict(templates.iloc[np.random.randint(0, len(templates))].copy(deep=True))
        template = res.pop("template")
        res["name"] = name
        component_words = self.template_to_ner(template, fill_values)
        component_words = label_entity(component_words, name)
        res["words"] = component_words
        res["text"] = " ".join([ner_word.value for ner_word in component_words])

        return res

    def get_fill_values(self, component):
        """
        Sometimes values from bet components restrict the values that can be used in later components of the bet.
        e.g. For the event 'Real Madrid vs Barcelona', the only teams that can be chosen are 'Real Madrid' and 'Barcelona'.
        """
        entities = ner_words_to_entities(component["words"])
        fill_values = {"TEAM": [" ".join(entity["text"]) for entity in entities if entity["name"] == "TEAM"]}

        return fill_values

    def make_bet(self):
        bet = {}
        bet["event"] = self.make_component("EVENT", self.templates["event"])
        fill_values = self.get_fill_values(bet["event"])
        # many valid market templates for a given event
        bet["market"] = self.make_component(
            "MARKET", self.templates["market"][self.templates["market"]["event_id"] == bet["event"]["id"]], fill_values
        )
        # one valid side template for a given market
        side_id = bet["market"]["side_id"]
        bet["side"] = self.make_component("SIDE", self.templates["side"][self.templates["side"]["id"] == side_id], fill_values)
        bet["odds"] = self.make_component("ODDS", self.templates["odds"], fill_values)

        return bet
