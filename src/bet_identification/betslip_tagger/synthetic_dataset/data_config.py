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
    coarse_ner_labels: list = [
        "O",
        "B-EVENT",
        "I-EVENT",
        "B-MARKET",
        "I-MARKET",
        "B-SIDE",
        "I-SIDE",
        "B-ODDS",
        "I-ODDS",
    ]
    fine_ner_labels: list = [
        "B-VERSUS",
        "I-VERSUS",
        "B-TEAM",
        "I-TEAM",
        "B-PLAYER",
        "I-PLAYER",
        "B-SCORELINE",
        "I-SCORELINE",
        "B-BOOLEAN",
        "I-BOOLEAN",
        "B-PLUS_MINUS",
        "I-PLUS_MINUS",
    ]


data_config = DataConfig()
