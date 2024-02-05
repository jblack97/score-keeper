import numpy as np

from data_generation import BetMaker


class SyntheticBetslipGenerator:
    def __init__(self, bet_maker: BetMaker) -> None:
        self.bet_maker = bet_maker

    def generate_betslip(self):
        num_bets = np.random.randint(0, 5)
        lines = []
        # Betslips can have irrelevant text at the top
        for _ in range(np.random.randint(0, 5)):
            line = []
            for _ in range(np.random.ranint(1, 10)):
                line.append(self.bet_maker.make_component("NOISE", self.bet_maker.templates["noise"]))
            lines.append(line)
        for bet in range(num_bets):
            bet = self.bet_maker.make_bet()
            # sometimes want to inject noise line
            # sometimes want to inject noise characters
            # TODO index BET with component NAMES, not ints
            event_at_start = False
            if np.random.random() > 0.50:
                lines.append([bet["event"]])
                event_at_start = True

            if np.random.random() > 0.85:
                lines.append([self.bet_maker.make_component("NOISE", self.bet_maker.templates["noise"])])

            lines.append([bet["market"]])

            if np.random.random() > 0.85:
                lines.append([self.bet_maker.make_component("NOISE", self.bet_maker.templates["noise"])])

            line = []
            line.append(bet["side"])
            line.append(bet["odds"])
            lines.append(line)
            if not event_at_start:
                lines.append([bet["event"]])

        # Betslips can have lots of irrelevant text at the bottom
        for _ in range(np.random.randint(0, 5)):
            lines.append([self.bet_maker.make_component("NOISE", self.bet_maker.templates["noise"])])

        return lines
