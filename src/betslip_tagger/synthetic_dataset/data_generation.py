from abc import ABC, abstractmethod
import numpy as np


class Maker(ABC):
    def __init__(self, templates, values):
        self.templates = templates
        self.values = values

    @staticmethod
    def get_random_value(values):
        num = np.random.randint(len(values))

        return values[num]

    @abstractmethod
    def make(self):
        pass


class EventMaker(Maker):
    def __init__(self, templates, values):
        self.templates = templates
        self.values = values

    def make(self):
        # randomly choose template
        # for each word in template, check if a placeholder
        # fill placeholders
        template = self.get_random_value(self.templates).split()
        for ind, word in enumerate(template):
            if word in placeholders:
                template[ind] = self.fill_placeholder(word)

        return template

    def fill_placeholder(self):
        pass


class OddsMaker(Maker):
    def __init__(self):
        pass


class MarketMaker(Maker):
    def __init__(self):
        pass


class SideMaker(Maker):
    def __init__(self):
        pass


class BetMaker:
    def __init__(self, event_maker, market_maker, side_maker, odds_maker):
        self.event_maker = event_maker
        self.market_maker = market_maker
        self.side_maker = side_maker
        self.odds_maker = odds_maker

    def make_bet(self):
        bet = {}
        bet["event"] = self.event_maker.make_event()
        bet["market"] = self.market_maker.make_market(bet["event"].type)
        bet["side"] = self.side_maker.make_side(bet["market"].type)
        bet["odds"] = self.odds_maker.make_odds()
