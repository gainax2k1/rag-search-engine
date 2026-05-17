from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.search_utils import cosine_similarity, load_movies

MODEL = "clip-ViT-B-32"

class MultimodalSearch:
    def __init__(self, model_name=MODEL, docs=None):
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs] if docs else None
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True) if self.texts else None

    def encode_text(self, text):
        return self.model.encode(text)

    def encode_image(self, image_path):
        image = Image.open(image_path)
        return self.model.encode(image)
    
    def search_with_image(self, image_path, limit=5):
        image_embedding = self.encode_image(image_path)
        scores = []
        for idx, text_embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            scores.append((score, self.docs[idx]))
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

        return sorted_scores[:limit]
    
def verify_image_embedding(image_path):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.encode_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path, limit=5):
    documents = load_movies()
    multimodal_search = MultimodalSearch(docs=documents)
    results = multimodal_search.search_with_image(image_path, limit)
    print("Search Results:")
    idx = 1
    for score, doc in results:
        print(f"\n{idx}. {doc['title']} (similarity: {score:.3f})")
        print(f"   {doc['description'][:200]}...")
        idx += 1