import os
import json
import openai
from pathlib import Path
from config import config

ROOT = Path(__file__).parent


def make_default_message(role, content):
    return {"role": role, "content": content}


def make_messages(inference_data, prompt_files):
    messages = []
    for prompt_file in prompt_files:
        with open(ROOT / prompt_file, "r") as f:
            messages.append(make_default_message("user", f.read()))
    messages.append(make_default_message("user", inference_data))

    return messages


def gpt_text_to_bet(text_sets, prompt_dir):
    messages = []
    for prompt_file in os.listdir(prompt_dir):
        with open(ROOT / prompt_dir / prompt_file, "r") as f:
            messages.append(make_default_message("user", f.read()))
    messages.append(make_default_message("user", str(text_sets)))
    openai.api_key = config.openai_api_key
    res = openai.ChatCompletion.create(
        model=config.gpt_model_name,
        messages=messages,
        temperature=0.6,
    )
    bets = json.loads(res.to_dict()["choices"][0]["message"]["content"])

    return bets
