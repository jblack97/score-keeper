import os
import sys
import datetime
import pandas as pd
import argparse
import cv2
import json
from pathlib import Path
from src.betslip_reader.line_identification import identify_lines
from src.betslip_reader.text_recognition import apply_tess
from src.bet_identification.text_to_bet import gpt_text_to_bet

ROOT = Path(__file__).parent.parent


def parse_args(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--image_dir",
        help="Directory with images for inference.",
        default=ROOT / "betslips/images",
    )
    arg_parser.add_argument(
        "--input_config",
        help="Config with names of images",
        default=ROOT / "inference_outputs/inference_images.csv",
    )
    arg_parser.add_argument(
        "--save_lines",
        help="Whether to save the individual line jpg files.",
        default=False,
    )
    arg_parser.add_argument(
        "--output_dir",
        help="Directory where outputs are written.",
        default=ROOT / "inference_outputs",
    )
    arg_parser.add_argument(
        "--prompt_dir",
        help="Directory containing GPT prompt",
        default=ROOT / "src/prompts/text_to_bet",
    )
    args = arg_parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    run_timestamp = datetime.datetime.now().strftime("%d%m%y%H%M%S")
    os.makedirs(f"{args.output_dir}/{run_timestamp}")
    text_sets = []
    image_ids = pd.read_csv(args.input_config)["image_id"]
    for image_id in image_ids:
        # get line images
        line_images = identify_lines(f"{args.image_dir}/{image_id}/betslip.jpg")
        lines_dir = f"{args.output_dir}/{run_timestamp}/{image_id}/lines"
        if not os.path.exists(lines_dir):
            os.makedirs(lines_dir)
        for ind, image in enumerate(line_images):
            image_filepath = f"{lines_dir}/line_{ind}.jpg"
            print(f"Saving line to {image_filepath}")
            cv2.imwrite(image_filepath, image)

        text_sets.append(apply_tess(lines_dir))

    bets = gpt_text_to_bet(text_sets, args.prompt_dir)
    output = []
    text_sets = pd.Series(text_sets)
    for bet_ind, bet_name in enumerate(bets.keys()):
        bet_output = {}
        # assumes gpt won't mess up number of images
        bet_output["image_id"] = image_ids.get(bet_ind, None)
        bet_output["text"] = text_sets.get(bet_ind, None)
        bet_output["bets"] = bets[bet_name]
        output.append(bet_output)

    with open(f"{args.output_dir}/{run_timestamp}/output.json", "w") as f:
        f.write(json.dumps(output))


if __name__ == "__main__":
    main(sys.argv[1:])
