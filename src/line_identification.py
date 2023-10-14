import sys
import cv2
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN


def show_contours(image, contours, colours):
    for _, row in contours.iterrows():
        if row["label"] == -1:
            continue
        x, y, w, h = cv2.boundingRect(row["contour"])
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            tuple(int(x) for x in colours[row["label"]]),
            2,
        )
        cv2.imshow("image", rect)
        cv2.waitKey(0)


def collect_contour_info(contours):
    contour_info = []
    for _, cnt in enumerate(contours):
        cnt_dic = {}
        cnt_dic["left"], cnt_dic["top"], w, h = cv2.boundingRect(cnt)
        cnt_dic["contour"] = cnt
        cnt_dic["bottom"] = cnt_dic["top"] + h
        cnt_dic["right"] = cnt_dic["left"] + w

        contour_info.append(cnt_dic)
    df = pd.DataFrame(contour_info).sort_values(by="top", ascending=True)

    return df


def create_colour_scheme(label_count):
    colour_scheme = {-1: np.array([0, 0, 0])}
    colours = np.random.randint(0, 255, size=(label_count - 1, 3), dtype=int)
    for ind, colour in enumerate(colours):
        colour_scheme[ind] = tuple(int(x) for x in colour)

    return colour_scheme


def cluster_contours(df):
    """
    Uses DBSCAN to cluster rectangular contours around words into lines of text.
    """
    dbscanner = DBSCAN(np.median(df["bottom"] - df["top"]) / 2)
    df["label"] = dbscanner.fit(df[["top", "bottom"]]).labels_

    lines = (
        df.groupby("label")[["top", "bottom", "left", "right"]]
        .agg({"top": "min", "bottom": "max", "left": "min", "right": "max"})
        .drop(-1)
    )

    return lines


def cut_out_lines(img, lines):
    line_images = []
    for _, line in lines.iterrows():
        # extend edges of lines with padding to improve text recognition
        pad = 4
        line_images.append(
            img[
                line["top"] - pad : line["bottom"] + pad,
                line["left"] - pad : line["right"] + pad,
            ]
        )

    return line_images


def identify_lines(image_path):
    """
    Identifies lines of text in betslip by first identifying rectangular contours around pieces
    of text and then clustering them into lines using DBSCAN.
    Saves individual line images to a subdirectory of the image.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    df = collect_contour_info(contours)
    lines = cluster_contours(df)
    line_images = cut_out_lines(img, lines)

    return line_images
