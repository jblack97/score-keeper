import numpy as np

from bet_maker import BetMaker


class SyntheticBetslipMaker:
    def __init__(self, bet_maker: BetMaker) -> None:
        self.bet_maker = bet_maker

    def generate_betslip(self):
        num_bets = np.random.randint(0, 6)
        lines = []
        # Betslips can have irrelevant text at the top
        for _ in range(np.random.randint(0, 3)):
            line = []
            for _ in range(np.random.randint(1, 3)):
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
