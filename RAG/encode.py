# Encode function to encode documents into embeddings space

import re
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-mpnet-base-v2')

def preprocess(document: str) -> str:
    document = document.lower().replace('\n', ' ').replace('\t', ' ')
    document = re.sub(r"[^a-z0-9\s\.,!?'-]", '', document)
    document = re.sub(r'\s+', ' ', document).strip()
    return document

def encode(document):
    document = preprocess(document)
    embeded_document = embedding_model.encode(document)
    return embeded_document

def normalize(embedding):
    norm = sum([x**2 for x in embedding]) ** 0.5
    normalized_embedding = [x / norm for x in embedding]
    return normalized_embedding

if __name__ == "__main__":
    dummy_text = """
    Capture every moment with stunning clarity and detail using the Canon EOS 70D DSLR. Designed for both passionate enthusiasts and advanced photographers, the 70D features a 20.2-megapixel APS-C CMOS sensor combined with Canon's DIGIC 5+ image processor to deliver crisp, vibrant images and smooth performance in virtually any lighting condition.

    The camera's Dual Pixel CMOS Autofocus system ensures fast, accurate focus in live view and video recording, making it perfect for both photography and full HD 1080p video. Its 19-point all cross-type AF system provides precise focusing in challenging shooting environments.

    Enjoy creative flexibility with the fully articulating 3-inch touchscreen LCD, which allows intuitive framing, focusing, and menu navigation. Built-in Wi-Fi enables seamless sharing and remote shooting via compatible devices, while continuous shooting at 7 frames per second ensures you never miss a fast-moving subject.

    Compact yet powerful, the Canon EOS 70D is a versatile DSLR that balances professional-grade performance with user-friendly controls, making it an ideal choice for capturing everything from everyday moments to high-quality artistic photography.
    """

    preprocessed_text = preprocess(dummy_text)
    processed_text = encode(dummy_text)
    processed_text = normalize(processed_text)
    print(processed_text)