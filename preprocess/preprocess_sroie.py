import os
import json
import tqdm
from dataclasses import dataclass

@dataclass
class PreprocessorSROIE:
    processed_data: dict = {}
    process_mode: str = 'train'
    img_path: str = 'img'
    entities_path: str = 'entities'
    box_path: str = 'box'
    lines: list = []

    def __post_init__(self):
        if self.process_mode == 'train':
            self.base_path = '/home/ishan/Downloads/archive/SROIE2019/train/'
        else:
            self.base_path = '/home/ishan/Downloads/archive/SROIE2019/test/'
        
        self.base_files = []
        assert len(os.listdir(self.base_path+self.img_path)) == len(os.listdir(self.base_path+self.entities_path)) == len(os.listdir(self.base_path+self.box_path))
        for file in os.listdir(self.base_path+self.img_path):
            self.base_files.append(file.split('.')[0])

    def process_bbox(self, path):
        # Split Bounding box coordinates for each word
        data = open(path, 'r').readlines()
        lines = []
        for line in data:
            line = line.strip('\n').strip().split(',')
            text = ",".join(line[8:])

            per_word_width = (int(line[6]) - int(line[0]))/len(text)
            words = text.split(' ')
            word_pointer = line[0]
            line_words = []
            for word in words:
                self.word_data.append(word)
                self.bbox_data.append([word_pointer, line[1], word_pointer+per_word_width*len(word), line[7]])
                line_words.append([self.bbox_data[-1], self.word_data[-1]])
                word_pointer += per_word_width*len(word)+per_word_width
            lines.append(line_words)
        return lines

    def process_entities(self, path, lines):
        # Add bounding box coordinates to entities
        data = json.loads(open(path, 'r').read())
        labels = []
        for k, v in data.items():
            for i in range(len(v)):
                v[i]['box'] = self.lines[int(k)][i][0]
                v[i]['word'] = self.lines[int(k)][i][1]
        pass

    def process(self):
        for file in self.base_files:
            lines = self.process_bbox(self.base_path+self.box_path+file+'.txt')
            self.process_entities(self.base_path+self.entities_path+file+'.txt', lines)
            if self.process_mode == 'train':
                self.processed_data[file] = {
                    'img': self.base_path+self.img_path+file+'.jpg',
                    'box': self.bbox_data,
                    'words': self.word_data,
                    'labels': self.entities_data
                }
            else:
                self.processed_data[file] = {
                    'img': self.base_path+self.img_path+file+'.jpg',
                    'box': self.bbox_data,
                    'words': self.word_data,
                }
        open('processed_data.json', 'w').write(self.processed_data)
        return self.processed_data