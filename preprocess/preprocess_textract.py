import io
import boto3
import os
import dotenv

dotenv.load_dotenv()

class PreProcessTextract:
    def __init__(self,
                 process_mode: str = 'train',
                 img_path: str = 'raw-data/imgs/',
                 tag_path: str = 'raw-data/tags/'):

        self.process_mode = process_mode
        self.img_path = img_path
        self.tag_path = tag_path
        self.s3client = boto3.client('s3',
                                   aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                   aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        self.textractclient = boto3.client('textract',
                                            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
        self.bucket_name = os.environ['AWS_BUCKET_NAME']

        self.processed_data: list = []
        self.word_data: list = []
        self.box_data: list = []
        self.entities_data: list = []

    def get_file_from_bucket(self, file_name):
        file_obj = self.s3client.download_file(self.bucket_name, self.img_path+file_name)
        return file_obj

    def process_text(self, file_object):
        textract_response = self.textractclient.analyze_document(
            Document={
                'Bytes': file_object
            },
            FeatureTypes=[
                'TEXT'
            ]
        )
        # process for bbox, text, entities and give 'O' tag to all
        return textract_response


    def get_entites_from_bbox(self, bbox, textract_response):
        # change tags based on bbox
        pass

    def normalize_bbox(self):
        # scale to 1000
        pass

    def upload_data_to_bucket(self):
        pass

    def process(self):
        pass

if __name__ == "__main__":
    preprocess = PreProcessTextract()
    preprocess.process()