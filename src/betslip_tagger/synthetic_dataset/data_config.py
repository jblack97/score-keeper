import json
from pydantic_settings import BaseSettings
from .placeholders import LookUpFiller, PlayerFiller, IntFiller, FloatFiller, ScorelineFiller


class DataConfig(BaseSettings):
    entities: list = ["event", "market", "side", "odds"]
    placeholders: dict = {
        "VERSUS": {"filler": LookUpFiller},
        "TEAM": {"filler": LookUpFiller},
        "PLAYER": {"filler": PlayerFiller},
        "INT": {"filler": IntFiller},
        "FLOAT": {"filler": FloatFiller},
        "SCORELINE": {"filler": ScorelineFiller},
        # "COMPETITION": {"filler": LookUpFiller},
        # "SPONSOR": {"filler": "generate"},
    }


data_config = DataConfig()
