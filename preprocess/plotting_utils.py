import json
import argparse
import cv2
import numpy as np

class DataPlotter:
    def __init__(self, data_path, img_path):
        self.data_path = data_path
        self.img_path = img_path

    def plot_entities(self):
        annotations = json.loads(open(self.data_path, "r").read())
        for annot in annotations:
            if annot["imgs"].split('/')[-1] == self.img_path.split('/')[-1]:
                img = cv2.imread(self.img_path)
                resize_img = cv2.resize(img  , (1000,1000))
                for idx, label in enumerate(annot["labels"]):
                    if label != "OTHER":
                        annot["boxes"][idx] = np.rint(annot["boxes"][idx]).astype(int)
                        cv2.rectangle(resize_img, annot["boxes"][idx][:2],annot["boxes"][idx][2:], (0, 255, 0), 2)
                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Image", 750, 750)
                cv2.imshow("Image", resize_img)
                cv2.waitKey(0)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", type=str, required=True)
    arg_parser.add_argument("--img_path", type=str, required=True)

    args = arg_parser.parse_args()

    plotter = DataPlotter(args.data_path, args.img_path)
    plotter.plot_entities()


