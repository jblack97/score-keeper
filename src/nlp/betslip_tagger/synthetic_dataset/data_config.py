from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    bet_components: list = ["event", "market", "side", "odds", "noise"]
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
        "DATE": {"fill_by": "generate"},
        "RANDOM_WORD": {"fill_by": "lookup"},
        # "COMPETITION": {"fill_by": 'lookup'},
        # "SPONSOR": {"fill_by": "generate"},
    }

    ner_labels: dict = {
        "O": {"type": "coarse"},
        "B-EVENT": {"type": "coarse"},
        "I-EVENT": {"type": "coarse"},
        "B-MARKET": {"type": "coarse"},
        "I-MARKET": {"type": "coarse"},
        "B-SIDE": {"type": "coarse"},
        "I-SIDE": {"type": "coarse"},
        "B-ODDS": {"type": "coarse"},
        "I-ODDS": {"type": "coarse"},
        "B-VERSUS": {"type": "fine"},
        "I-VERSUS": {"type": "fine"},
        "B-TEAM": {"type": "fine"},
        "I-TEAM": {"type": "fine"},
        "B-PLAYER": {"type": "fine"},
        "I-PLAYER": {"type": "fine"},
        "B-SCORELINE": {"type": "fine"},
        "I-SCORELINE": {"type": "fine"},
        "B-BOOLEAN": {"type": "fine"},
        "I-BOOLEAN": {"type": "fine"},
        "B-PLUS_MINUS": {"type": "fine"},
        "I-PLUS_MINUS": {"type": "fine"},
        "B-DATE": {"type": "fine"},
        "I-DATE": {"type": "fine"},
    }


data_config = DataConfig()
