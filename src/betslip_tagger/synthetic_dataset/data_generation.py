import numpy as np
import pandas as pd


class Maker:
    def __init__(self, name, fillers: dict):
        self.name = name
        self.fillers = fillers

    def make(self, templates: pd.DataFrame):
        ind = np.random.randint(0, len(templates))
        res = dict(templates.iloc[ind].copy(deep=True))
        res["name"] = self.name
        template = res.pop("template")
        for ind, word in enumerate(template):
            if word in self.fillers:
                template[ind] = self.fillers[word]["filler"].fill()
        res["value"] = template

        return res


class EventMaker(Maker):
    def __init__(self, templates, values, fillers):
        super().__init__(self, "event", templates, values, fillers)


class OddsMaker(Maker):
    def __init__(self, templates, values, fillers):
        super().__init__(self, "odds", templates, values, fillers)


class MarketMaker(Maker):
    def __init__(self, templates, values, fillers):
        super().__init__(self, "market", templates, values, fillers)


class SideMaker(Maker):
    def __init__(self, templates, values, fillers):
        super().__init__(self, "sides", templates, values, fillers)


class BetMaker:
    def __init__(self, event_maker, market_maker, side_maker, odds_maker, templates):
        self.event_maker = event_maker
        self.market_maker = market_maker
        self.side_maker = side_maker
        self.odds_maker = odds_maker
        self.templates = templates

    def make_bet(self):
        bet = {}
        bet["event"] = self.event_maker.make(self.templates["event"])
        # many valid market templates for a given event
        bet["market"] = self.market_maker.make(
            self.templates["event"][self.templates["event"]["event_id"] == bet["event"]["id"]]
        )
        # one valid side template for a given market
        side_id = self.templates["market"][self.templates["market"]["market"] == bet["market"]["value"]]["side_id"]
        bet["side"] = self.side_maker.make(self.templates["side"][self.templates["side"]["id"] == side_id])
        bet["odds"] = self.odds_maker.make(self.templates["odds"])

        return bet
