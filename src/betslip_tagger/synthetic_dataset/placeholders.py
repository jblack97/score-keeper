import json
import numpy as np
import names


class PlaceholderFiller:
    def __init__(self, placeholder):
        self.placeholder = placeholder


class LookUpFiller(PlaceholderFiller):
    def __init__(self, placeholder, lookup_dir):
        super().__init__(placeholder)
        with open(f"{lookup_dir}/{placeholder}.json", "r") as f:
            self.values = json.loads(f.read())

    def fill(self):
        return np.random.choice(self.values)["value"]


class PlayerFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("PLAYER")

    @staticmethod
    def fill():
        return names.get_full_name()


class IntFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("INT")

    @staticmethod
    def fill():
        return str(np.random.randint(0, 5))


class FloatFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("FLOAT")

    @staticmethod
    def fill():
        return str(np.around(np.random.randint(0, 100) * np.random.random(), 2))


class ScorelineFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("SCORELINE")

    @staticmethod
    def fill():
        return f"{np.random.randint(0,8)} - {np.random.randint(0,8)}"


class FillerMaker:
    def generate_filler(placeholder, kind=None, lookup_dir=None):
        if kind == "lookup":
            return LookUpFiller(placeholder, lookup_dir=lookup_dir)
        if placeholder == "PLAYER":
            return PlayerFiller()
        if placeholder == "INT":
            return IntFiller()
        if placeholder == "FLOAT":
            return FloatFiller()
        if placeholder == "SCORELINE":
            return ScorelineFiller()
        else:
            raise ValueError(f"No filler implemented for : {placeholder}")
