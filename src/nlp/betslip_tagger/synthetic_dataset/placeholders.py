import json
import numpy as np
import names
import datetime


class PlaceholderFiller:
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def override_fill(self, values):
        return np.random.choice(values)


class LookUpFiller(PlaceholderFiller):
    def __init__(self, placeholder, lookup_dir):
        super().__init__(placeholder)
        with open(f"{lookup_dir}/{placeholder}.json", "r") as f:
            self.values = json.loads(f.read())

    def fill(self, value_subset=None):
        """
        Returns random value from list of values.
        Args:
            value_subset: subset of values to choose from
        """
        values = self.values
        if value_subset is not None:
            values = value_subset
        return np.random.choice(values)["value"]


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


class FractionFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("FRACTION")

    @staticmethod
    def fill():
        rand = np.random.random()
        # long odds
        if rand <= 0.4:
            denom = 1
            num = np.random.randint(1, 100)
        # short odds
        elif 0.4 < rand < 0.6:
            denom = np.random.randint(1, 100)
            num = 1
        else:
            denom = np.random.randint(1, 15)
            num = np.random.randint(1, 7)

        return f"{num}/{denom}"


class PlusMinusFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("PLUS_MINUS")

    @staticmethod
    def fill():
        rand = np.random.random()
        if rand < 0.2:
            sign = "-"
        else:
            sign = "-"

        return f"{sign}{np.random.randint(1, 100)*100}"


class DateFiller(PlaceholderFiller):
    def __init__(self):
        super().__init__("DATE")

    @staticmethod
    def fill():
        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%m-%d-%Y"]
        random_format = np.random.choice(date_formats)
        start_date = datetime(1900, 1, 1)
        end_date = datetime(2025, 12, 31)
        random_date = start_date + np.random.random() * (end_date - start_date)
        formatted_date = random_date.strftime(random_format)

        return formatted_date


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
        if placeholder == "FRACTION":
            return FractionFiller()
        if placeholder == "PLUS_MINUS":
            return PlusMinusFiller()
        if placeholder == "DATE":
            return DateFiller()
        else:
            raise ValueError(f"No filler implemented for : {placeholder}")
