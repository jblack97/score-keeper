import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract
from PIL import Image


def apply_tess(line_dir):
    line_images = [
        Image.open(f"{line_dir}/{line_path}").convert("RGB")
        for line_path in os.listdir(line_dir)
    ]
    image_text = []
    for line_image in line_images:
        line_text = pytesseract.image_to_string(line_image, config="--psm 7")
        print(line_text)
        image_text.append(line_text)

    return image_text


def apply_trocr(line_dir):
    line_images = [
        Image.open(line_path).convert("RGB") for line_path in os.listdir(line_dir)
    ]
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    image_text = []
    for line_image in line_images:
        pixel_values = processor(line_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(line_text)
        image_text.append(line_text)
