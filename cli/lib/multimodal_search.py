from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL = "clip-ViT-B-32"

class MultimodalSearch:
    def __init__(self, model_name=MODEL):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        return self.model.encode(text)

    def encode_image(self, image_path):
        image = Image.open(image_path)
        # Here we would have some logic to convert the image into a format suitable for the model
        # For example, we might use a pre-trained vision model to extract features from the image
        # and then combine those features with the text embeddings in some way.
        # This is a placeholder for demonstration purposes.
        return self.model.encode(image)
    
def verify_image_embedding(image_path):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.encode_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")