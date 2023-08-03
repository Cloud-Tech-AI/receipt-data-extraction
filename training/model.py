from transformers import AutoProcessor, AutoModelForTokenClassification


class Model:
    def __init__(self, use_large=False):
        if use_large:
            self.model = "microsoft/layoutlmv2-large-uncased"
            self.processor = "microsoft/layoutlmv2-large-uncased"
        else:
            self.model = "microsoft/layoutlmv2-base-uncased"
            self.processor = "microsoft/layoutlmv2-base-uncased"

    def load_model(self):
        return AutoModelForTokenClassification.from_pretrained(self.model)

    def load_processor(self):
        return AutoProcessor.from_pretrained(self.processor, revision="no_ocr")

    def save_model():
        pass
