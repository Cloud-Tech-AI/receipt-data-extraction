from tqdm import tqdm
import boto3
import os
import io
import json
import dotenv
from PIL import Image
from shapely.geometry import box
import botocore


dotenv.load_dotenv()

class PreProcessTextract:
    def __init__(self,
                 process_mode: str = 'train',
                 img_path: str = 'raw-data/imgs/',
                 tag_path: str = 'raw-data/tags/',
                 upload_path: str = 'preprocessed-data-train/processed_data.json'):

        self.process_mode = process_mode
        self.img_path = img_path
        self.tag_path = tag_path
        try:
            response = self.s3client.get_object(Bucket=self.bucket_name, Key=upload_path)['Body'].read()
            self.target_data = json.loads(response)
        except botocore.exceptions.ClientError as e:
            self.target_data = []
        self.s3client = boto3.client('s3',
                                #    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                #    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
                                   )
        self.textractclient = boto3.client('textract',
                                           region_name='ap-south-1',
                                            # aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                            # aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
                                            )
        self.bucket_name = os.environ['AWS_BUCKET_NAME']

        self.processed_data: list = []
        self.word_data: list = []
        self.box_data: list = []
        self.entities_data: list = []

    def get_file_list(self, mode):
        file_list = self.s3client.list_objects(Bucket=self.bucket_name, Prefix=self.img_path)
        file_list = [file for file in file_list['Contents'] if file['Key'].split('.')[-1] == 'jpg']
        return file_list

    def get_file_from_bucket(self, file_name, type):
        if type == 'img':
            file_obj = self.s3client
            file_obj = self.s3client.get_object(Bucket=self.bucket_name, Key=file_name)['Body'].read()
        else:
            file_name = file_name.replace('imgs','tags').replace('jpg', 'json')
            file_obj = self.s3client.get_object(Bucket=self.bucket_name, Key=file_name)['Body'].read()
        return file_obj

    def process_text(self, img, tag):
        textract_response = self.textractclient.detect_document_text(
            Document={'Bytes': img}
        )
        image = Image.open(io.BytesIO(img))
        width, height = image.size
        words =  textract_response['Blocks']
        words = [word for word in words if word['BlockType'] == 'WORD']
        self.word_data = [word['Text'] for word in words]
        self.box_data = [[word['Geometry']['Polygon'][0]['X']*width, word['Geometry']['Polygon'][0]['Y']*height,
                          word['Geometry']['Polygon'][3]['X']*width, word['Geometry']['Polygon'][3]['Y']*height] for word in words]
        self.entities_data = ['O' for _ in words]
        self.get_entites_from_bbox(tag)
        self.box_data = [self.normalize_bbox(bbox, width, height) for bbox in self.box_data]

    def get_entites_from_bbox(self, tag):
        label_dict = {
            "COMPANY": 0,
            "DATE": 0,
            "ADDRESS": 0,
            "TOTAL": 0
        }
        tag = json.loads(tag)
        for k in tag:
            for bbox in tag[k]:
                for i in range(len(self.box_data)):
                    box1 = box(self.box_data[i][0], self.box_data[i][1], self.box_data[i][2], self.box_data[i][3])
                    box2 = box(bbox[0], bbox[1], bbox[2], bbox[3])
                    if box1.intersects(box2):
                        intersection_area = box1.intersection(box2).area
                        bbox1_area = box1.area
                        bbox2_area = box2.area
                        if bbox1_area == 0 or bbox2_area == 0:
                            continue
                        intersection_percentage = (intersection_area / min(bbox1_area , bbox2_area)) * 100
                        if intersection_percentage > 85:
                            if label_dict[k.upper()] == 0:
                                self.entities_data[i] = 'B-'+k.upper()
                                label_dict[k.upper()] += 1
                            else:
                                self.entities_data[i] = 'I-'+k.upper()

    def normalize_bbox(self, bbox, width, height):
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def upload_data_to_bucket(self):
        self.target_data.extend(self.processed_data)
        self.s3client.put_object(Bucket=self.bucket_name, Key=self.upload_path, Body=json.dumps(self.target_data))

    def process(self):
        all_files = self.get_file_list('process')
        for file in tqdm(all_files):
            if file['Key'].split('/')[-1] in self.target_data:
                continue
            img = self.get_file_from_bucket(file['Key'], 'img')
            tag = self.get_file_from_bucket(file['Key'], 'tag')
            self.process_text(img, tag)
            self.processed_data.append({
                'imgs': file['Key'],
                'boxes': self.box_data,
                'words': self.word_data,
                'labels': self.entities_data
            })
            self.box_data = []
            self.word_data = []
            self.entities_data = []
        self.upload_data_to_bucket()

if __name__ == "__main__":
    preprocess = PreProcessTextract()
    preprocess.process()