import sys
import os
import cv2
import argparse
import json
import uuid


def parse_args(argv):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input_image_dir",
        help="directory where betslip images are stored",
        default="/Users/jblack/projects/score_keeper/betslips/triage",
    )
    arg_parser.add_argument(
        "--output_image_dir",
        help="directory where betslip images are stored",
        default="/Users/jblack/projects/score_keeper/betslips/images",
    )
    args = arg_parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for image_file in os.listdir(args.input_image_dir):
        image_path = f"{args.input_image_dir}/{image_file}"
        id = uuid.uuid1()
        metadata = {"id": str(id)}
        image = cv2.imread(image_path)
        new_image_dir = f"{args.output_image_dir}/{id}"
        os.mkdir(new_image_dir)
        cv2.imwrite(f"{new_image_dir}/betslip.jpg", image)
        with open(f"{new_image_dir}/metadata.json", "w") as f:
            f.write(json.dumps(metadata))
        os.remove(image_path)


if __name__ == "__main__":
    main(sys.argv[1:])
