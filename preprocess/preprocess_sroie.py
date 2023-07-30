import os

class PreprocessorSROIE:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data = []
        self.data_processed = []
        self.data_processed_df = N
    
    def read_data(self):
        for file in os.listdir(self.data_dir):
            if file.endswith(".txt"):
                self.data.append(os.path.join(self.data_dir, file))