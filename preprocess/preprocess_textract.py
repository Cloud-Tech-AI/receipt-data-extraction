import os
import io
import json
from dataclasses import dataclass, field
from typing import List
import botocore
import boto3
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

from data_models import Word, Bbox
from constants import LABEL_DICT


@dataclass
class PreProcessTextract:
    """Transform Textract generated OCR data into a format suitable for training a model."""

    process_mode: str = "train"
    img_path: str = "raw-data/imgs/"
    tag_path: str = "raw-data/tags/"
    infer_path: str = "user-data/imgs/"
    upload_path: str = "preprocessed-data-train/processed_data.json"
    s3client: boto3.client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    textractclient: boto3.client = boto3.client(
        "textract",
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    bucket_name: str = "receipt-extraction-data"

    processed_data: list = field(default_factory=list)
    word_data: list = field(default_factory=list)
    box_data: list = field(default_factory=list)
    entities_data: list = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            response = self.s3client.get_object(
                Bucket=self.bucket_name, Key=self.upload_path
            )["Body"].read()
            self.target_data = json.loads(response)
        except botocore.exceptions.ClientError as e:
            self.target_data = []

    def get_file_list(self) -> List[str]:
        """Get list of files in the bucket

        Returns:
            List[str]: List of file names
        """
        file_list = self.s3client.list_objects(
            Bucket=self.bucket_name, Prefix=self.img_path
        )
        file_list = [
            file
            for file in file_list["Contents"]
            if file["Key"].split(".")[-1] == "jpg"
        ]
        return file_list

    def get_file_from_bucket(self, file_name, type):
        """Get file from the bucket"""
        if type == "img":
            file_obj = self.s3client
            file_obj = self.s3client.get_object(Bucket=self.bucket_name, Key=file_name)[
                "Body"
            ].read()
        else:
            file_name = file_name.replace("imgs", "tags").replace("jpg", "json")
            file_obj = self.s3client.get_object(Bucket=self.bucket_name, Key=file_name)[
                "Body"
            ].read()
        return file_obj

    def process_text(self, img, tag) -> None:
        """Create Data for training
        - Extract text from image using Textract and parse the response to get words and their bounding boxes
        - Add Label to OCR words based on the bounding box coordinates available in the tag file
        """
        textract_response = self.textractclient.detect_document_text(
            Document={"Bytes": img}
        )
        image = Image.open(io.BytesIO(img))
        width, height = image.size
        words = textract_response["Blocks"]
        words = [word for word in words if word["BlockType"] == "WORD"]
        words = [
            Word(
                Bbox(
                    word["Geometry"]["Polygon"][0]["X"] * width,
                    word["Geometry"]["Polygon"][0]["Y"] * height,
                    word["Geometry"]["Polygon"][3]["X"] * width,
                    word["Geometry"]["Polygon"][3]["Y"] * height,
                ),
                word["Text"],
                "O",
            )
            for word in words
        ]
        self.word_data = [word.text for word in words]
        self.box_data = [word.bbox for word in words]
        if tag:
            self.entities_data = ["O"] * len(words)
            self.get_entites_from_bbox(tag)
        self.box_data = [bbox.normalize_bbox(width, height) for bbox in self.box_data]

    def get_entites_from_bbox(self, tag=None) -> None:
        """Add Label to OCR words based on the bounding box coordinates available in the tag file"""
        tag = json.loads(tag)
        label_dict = deepcopy(LABEL_DICT)
        for k in tag:
            for bbox in tag[k]:
                bbox1 = Bbox.from_list(bbox)
                for idx, bbox2 in enumerate(self.box_data):
                    if bbox1.intersects(bbox2):
                        intersection_area = bbox1.intersection(bbox2).area
                        bbox1_area = bbox1.area
                        bbox2_area = bbox2.area
                        if bbox1_area == 0 or bbox2_area == 0:
                            continue
                        intersection_percentage = (
                            intersection_area / min(bbox1_area, bbox2_area)
                        ) * 100
                        if intersection_percentage > 85:
                            if label_dict[k.upper()] == 0:
                                self.entities_data[idx] = "B-" + k.upper()
                                label_dict[k.upper()] += 1
                            else:
                                self.entities_data[idx] = "I-" + k.upper()

    def upload_data_to_bucket(self) -> None:
        """Upload processed data to the bucket"""
        self.target_data.extend(self.processed_data)
        self.s3client.put_object(
            Bucket=self.bucket_name,
            Key=self.upload_path,
            Body=json.dumps(self.target_data),
        )

    def process(self) -> None:
        """PreProcess all images in the bucket"""
        if self.process_mode == "train":
            all_files = self.get_file_list()
            target_files = [file["imgs"] for file in self.target_data]
            for file in tqdm(all_files):
                if file["Key"] in target_files:
                    continue
                img = self.get_file_from_bucket(file["Key"], "img")
                tag = self.get_file_from_bucket(file["Key"], "tag")
                self.process_text(img, tag)
                self.processed_data.append(
                    {
                        "imgs": file["Key"],
                        "boxes": self.box_data,
                        "words": self.word_data,
                        "labels": self.entities_data,
                    }
                )
                self.box_data = []
                self.word_data = []
                self.entities_data = []
            self.upload_data_to_bucket()
        else:
            self.process_text(img)
            self.processed_data.append(
                {
                    "imgs": "",
                    "boxes": self.box_data,
                    "words": self.word_data,
                }
            )


if __name__ == "__main__":
    preprocess = PreProcessTextract()
    preprocess.process()
