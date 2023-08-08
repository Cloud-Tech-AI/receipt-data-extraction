import os
import json
from PIL import Image
from tqdm import tqdm
import jellyfish


class PreprocessorSROIE():
    def __init__(self,
                 process_mode: str = 'train',
                 img_path: str = 'img/',
                 entities_path: str = 'entities/',
                 box_path: str = 'box/'):

        self.process_mode = process_mode
        self.img_path = img_path
        self.entities_path = entities_path
        self.box_path = box_path

        self.processed_data: list = []
        self.word_data: list = []
        self.box_data: list = []
        self.entities_data: list = []

        if self.process_mode == 'train':
            self.base_path = '/home/ishan/vscode-workspace/receipt-data-extraction/data/train/'
        else:
            self.base_path = '/home/ishan/vscode-workspace/receipt-data-extraction/data/test/'

        self.base_files = []
        assert len(os.listdir(self.base_path+self.img_path)) == len(os.listdir(
            self.base_path+self.entities_path)) == len(os.listdir(self.base_path+self.box_path))
        for file in os.listdir(self.base_path+self.img_path):
            self.base_files.append(file.split('.')[0])

    def normalize_bbox(self, bbox, width, height):
        return [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]

    def process_box(self, path):
        # Split Bounding box coordinates for each word
        data = open(path, 'r').readlines()
        lines = []
        for line in data:
            line = line.strip('\n').strip().split(',')
            text = ",".join(line[8:])

            per_word_width = (int(line[6]) - int(line[0]))/len(text)
            words = text.split(' ')
            word_pointer = int(line[0])
            line_words = []
            for word in words:
                line_words.append([[word_pointer, int(
                    line[1]), word_pointer+per_word_width*len(word), int(line[7])], word, "OTHER"])
                word_pointer += per_word_width*len(word)+per_word_width
            lines.append(line_words)
        return lines

    def process_entities(self, path, lines):
        # Add bounding box coordinates to entities
        data = json.loads(open(path, 'r').read())
        for k, v in data.items():
            for line in lines:
                text = " ".join([word[1] for word in line])
                if k.upper() == "ADDRESS":
                    if text in v and len(line) > 2:
                        for word in line:
                            word[2] = k.upper()
                elif k.upper() == "COMPANY":
                    distance = jellyfish.jaro_distance(text, v)
                    if distance > 0.8 and len(line) > 2:
                        for word in line:
                            if word[1] in v:
                                word[2] = k.upper()
                else:
                    distance = jellyfish.jaro_distance(text, v)
                    if distance > 0.8:
                        for word in line:
                            if word[1] in v:
                                word[2] = k.upper()

    def get_processed_data(self, lines, img_path):
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        for line in lines:
            for word in line:
                self.box_data.append(self.normalize_bbox(word[0], width, height))
                self.word_data.append(word[1])
                self.entities_data.append(word[2])

    def process(self):
        for file in tqdm(self.base_files):
            img_path = self.base_path+self.img_path+file+'.jpg'
            lines = self.process_box(self.base_path+self.box_path+file+'.txt')
            self.process_entities(
                self.base_path+self.entities_path+file+'.txt', lines)
            self.get_processed_data(lines, img_path)
            if self.process_mode == 'train':
                self.processed_data.append({
                    'imgs': img_path,
                    'boxes': self.box_data,
                    'words': self.word_data,
                    'labels': self.entities_data
                })
            else:
                self.processed_data.append({
                    'imgs': self.base_path+self.img_path+file+'.jpg',
                    'boxes': self.box_data,
                    'words': self.word_data,
                })
            self.box_data = []
            self.word_data = []
            self.entities_data = []
        open('processed_data.json', 'w').write(json.dumps(self.processed_data))
        return self.processed_data


if __name__ == "__main__":
    preprocessor = PreprocessorSROIE()
    preprocessor.process()
    print("Done!")
