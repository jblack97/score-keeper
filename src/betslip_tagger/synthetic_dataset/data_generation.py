import numpy as np
import pandas as pd


class BetMaker:
    def __init__(self, templates: dict, fillers: dict):
        self.templates = templates
        self.fillers = fillers

    def make_component(self, name, templates: pd.DataFrame):
        ind = np.random.randint(0, len(templates))
        res = dict(templates.iloc[ind].copy(deep=True))
        res["name"] = name
        template = res.pop("template").split()
        for ind, word in enumerate(template):
            if word in self.fillers:
                template[ind] = self.fillers[word].fill()
        res["value"] = template

        return res

    def make_bet(self):
        bet = {}
        bet["event"] = self.make_component("event", self.templates["event"])
        # many valid market templates for a given event
        bet["market"] = self.make_component(
            "market", self.templates["market"][self.templates["market"]["event_id"] == bet["event"]["id"]]
        )
        # one valid side template for a given market
        # side_id = self.templates["market"][self.templates["market"]["market"] == bet["market"]["value"]]["side_id"]
        side_id = bet["market"]["side_id"]
        bet["side"] = self.make_component("side", self.templates["side"][self.templates["side"]["id"] == side_id])
        bet["odds"] = self.make_component("odds", self.templates["odds"])

        return bet
