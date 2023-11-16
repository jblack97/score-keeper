from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    bet_components: list = ["event", "market", "side", "odds"]
    placeholders: dict = {
        "VERSUS": {"fill_by": "lookup"},
        "TEAM": {"fill_by": "lookup"},
        "PLAYER": {"fill_by": "generate"},
        "INT": {"fill_by": "generate"},
        "FLOAT": {"fill_by": "generate"},
        "SCORELINE": {"fill_by": "generate"},
        "OVER_UNDER": {"fill_by": "lookup"},
        "BOOLEAN": {"fill_by": "lookup"},
        "FRACTION": {"fill_by": "generate"},
        "PLUS_MINUS": {"fill_by": "generate"},
        # "COMPETITION": {"fill_by": 'lookup'},
        # "SPONSOR": {"fill_by": "generate"},
    }


data_config = DataConfig()
