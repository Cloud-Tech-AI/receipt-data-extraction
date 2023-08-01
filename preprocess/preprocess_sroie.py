import os
import json
from dataclasses import dataclass, field

from tqdm import tqdm
import jellyfish

@dataclass
class PreprocessorSROIE:
    processed_data: dict = field(default_factory=dict)
    process_mode: str = 'train'
    img_path: str = 'img/'
    entities_path: str = 'entities/'
    box_path: str = 'box/'
    word_data: list = field(default_factory=list)
    box_data: list = field(default_factory=list)
    entities_data: list = field(default_factory=list)

    def __post_init__(self):
        if self.process_mode == 'train':
            self.base_path = '/home/ishan/vscode-workspace/receipt-data-extraction/data/train/'
        else:
            self.base_path = '/home/ishan/vscode-workspace/receipt-data-extraction/data/test/'
        
        self.base_files = []
        assert len(os.listdir(self.base_path+self.img_path)) == len(os.listdir(self.base_path+self.entities_path)) == len(os.listdir(self.base_path+self.box_path))
        for file in os.listdir(self.base_path+self.img_path):
            self.base_files.append(file.split('.')[0])

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
                line_words.append([[word_pointer, int(line[1]), word_pointer+per_word_width*len(word), int(line[7])], word, "OTHER"])
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
                    if text in v and len(line)>2:
                        for word in line:
                            word[2] = k.upper()
                elif k.upper() == "COMPANY":
                    distance = jellyfish.jaro_distance(text, v)
                    if distance > 0.8 and len(line)>2:
                        for word in line:
                            if word[1] in v:
                                word[2] = k.upper()
                else:
                    distance = jellyfish.jaro_distance(text, v)
                    if distance > 0.8:
                        for word in line:
                            if word[1] in v:
                                word[2] = k.upper()

    def get_processed_data(self,lines):
        for line in lines:
            for word in line:
                self.box_data.append(word[0])
                self.word_data.append(word[1])
                self.entities_data.append(word[2])

    def process(self):
        for file in tqdm(self.base_files):
            lines = self.process_box(self.base_path+self.box_path+file+'.txt')
            self.process_entities(self.base_path+self.entities_path+file+'.txt', lines)
            self.get_processed_data(lines)
            if self.process_mode == 'train':
                self.processed_data[file] = {
                    'img': self.base_path+self.img_path+file+'.jpg',
                    'box': self.box_data,
                    'words': self.word_data,
                    'labels': self.entities_data
                }
            else:
                self.processed_data[file] = {
                    'img': self.base_path+self.img_path+file+'.jpg',
                    'box': self.box_data,
                    'words': self.word_data,
                }
            self.box_data = []
            self.word_data = []
            self.entities_data = []
        # open('processed_data.json', 'w').write(json.dumps(self.processed_data))
        return self.processed_data

if __name__ == "__main__":
    preprocessor = PreprocessorSROIE()
    preprocessor.process()
    print("Done!")