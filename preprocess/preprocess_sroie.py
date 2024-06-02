import os
import json
from dataclasses import dataclass, field
from typing import List
from copy import deepcopy

from PIL import Image
from tqdm import tqdm
import jellyfish
from dotenv import load_dotenv

from data_models import Word, Bbox
from constants import LABEL_DICT


load_dotenv()


@dataclass
class PreprocessorSROIE:
    """Transform SROIE dataset into a format suitable for training a model."""

    process_mode: str = "train"
    img_path: str = "img/"
    entities_path: str = "entities/"
    box_path: str = "box/"
    processed_data: list = field(default_factory=list)
    processed_data: list = field(default_factory=list)
    word_data: list = field(default_factory=list)
    box_data: list = field(default_factory=list)
    entities_data: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.process_mode == "train":
            self.base_path = f"{os.environ['BASE_DATA_PATH']}/train/"
        else:
            self.base_path = f"{os.environ['BASE_DATA_PATH']}/test/"

        self.base_files = []
        assert (
            len(os.listdir(self.base_path + self.img_path))
            == len(os.listdir(self.base_path + self.entities_path))
            == len(os.listdir(self.base_path + self.box_path))
        )
        for file in os.listdir(self.base_path + self.img_path):
            self.base_files.append(file.split(".")[0])

    def process_box(self, path: str, width: int) -> List[List[Word]]:
        """Split bounding box coordinates for each word

        Args:
            path (str): path to the file containing bounding box coordinates
            width (int): width of the image from corresponding image file

        Returns:
            List: List of Lines, each line containing a list of words with their bounding box coordinates
        """
        data = open(path, "r").readlines()
        lines = []
        for line in data:
            line = line.strip("\n").strip().split(",")
            text = ",".join(line[8:])

            line_word = Word(
                Bbox(int(line[0]), int(line[1]), int(line[4]), int(line[7])), text
            )
            per_char_width = (line_word.bbox.xmax - line_word.bbox.xmin) / len(text)

            words = text.split(" ")
            word_pointer = line_word.bbox.xmin
            line_words = []

            for word in words:
                new_pointer = per_char_width * len(word) + per_char_width
                line_words.append(
                    Word(
                        Bbox(
                            word_pointer,
                            line_word.bbox.ymin,
                            (
                                word_pointer + new_pointer
                                if word_pointer + new_pointer < width
                                else width
                            ),
                            line_word.bbox.ymax,
                        ),
                        word,
                        "O",
                    )
                )
                word_pointer = (
                    word_pointer + new_pointer
                    if word_pointer + new_pointer < width
                    else width
                )
            lines.append(line_words)
        return lines

    def process_entities(self, path: str, lines: List[List[Word]]) -> None:
        """Add bounding box coordinates to entities

        Args:
            path (str): path to the file containing entities
            lines (List[List[Word]]): List of Lines, each line containing a list of words with their bounding box coordinates
        """
        data = json.loads(open(path, "r").read())
        for k, v in data.items():
            for line in lines:
                text = " ".join([word.text for word in line])
                if k.upper() == "ADDRESS":
                    if text in v and len(line) > 2:
                        for word in line:
                            word.label = k.upper()
                elif k.upper() == "COMPANY":
                    similarity = jellyfish.jaro_similarity(text, v)
                    if similarity < 0.8 and len(line) > 2:
                        for word in line:
                            if word.text in v:
                                word.label = k.upper()
                else:
                    similarity = jellyfish.jaro_similarity(text, v)
                    if similarity < 0.8:
                        for word in line:
                            if word.text in v:
                                word.label = k.upper()

    def get_processed_data(self, width: int, height: int, lines: List[List[Word]]):
        """Get processed data for each image
        - Add bounding box coordinates to processed data
        - Add words to processed data
        - Add labels to processed data (along with B- and I- prefixes)

        Args:
            width (int): width of the image from corresponding image file
            height (int): height of the image from corresponding image file
            lines (List[List[Word]]): List of Lines, each line containing a list of words with their bounding box coordinates
        """
        label_dict = deepcopy(LABEL_DICT)
        for line in lines:
            line = sorted(line, key=lambda x: x.bbox.xmin)
            for word in line:
                self.box_data.append(word.bbox.normalize_bbox(width, height).to_list())
                self.word_data.append(word.text)
                if word.label in label_dict:
                    if label_dict[word.label] == 0:
                        self.entities_data.append("B-" + word.label)
                        label_dict[word.label] = 1
                    else:
                        self.entities_data.append("I-" + word.label)
                else:
                    self.entities_data.append("O")

    def process(self):
        """PreProcess all images in the dataset"""
        for file in tqdm(self.base_files):
            img_path = self.base_path + self.img_path + file + ".jpg"
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            lines = self.process_box(
                self.base_path + self.box_path + file + ".txt", width
            )
            self.process_entities(
                self.base_path + self.entities_path + file + ".txt", lines
            )
            self.get_processed_data(width, height, lines)
            if self.process_mode == "train":
                self.processed_data.append(
                    {
                        "imgs": img_path,
                        "boxes": self.box_data,
                        "words": self.word_data,
                        "labels": self.entities_data,
                    }
                )
            else:
                self.processed_data.append(
                    {
                        "imgs": img_path,
                        "boxes": self.box_data,
                        "words": self.word_data,
                    }
                )
            self.box_data = []
            self.word_data = []
            self.entities_data = []

        return self.processed_data


if __name__ == "__main__":
    preprocessor = PreprocessorSROIE()
    processed_data = preprocessor.process()
    open("../data/processed_data.json", "w").write(json.dumps(processed_data))
