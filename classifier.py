# classifier.py
"""
predict_category(image: PIL.Image) -> "top" / "bottom" / "dress"

This module uses your existing feature extractor & dataset nearest-neighbour
(predict_product_type in main_code.py) so you don't need to train a separate
image classifier. It writes the received PIL.Image to a temporary file, re-uses
extract_uploaded_features(...) and predict_product_type(...), and maps the
predicted product_type_name to one of: top / bottom / dress.
"""

import tempfile
import os
from PIL import Image
import numpy as np

# import the existing functions from your main_code
# main_code must exist and expose extract_uploaded_features and predict_product_type
from main_code import extract_uploaded_features, predict_product_type

# keywords to map dataset product_type_name -> category
TOP_KEYWORDS = ['shirt', 't-shirt', 'top', 'blouse', 'tee']
BOTTOM_KEYWORDS = ['pant', 'jean', 'trouser', 'short', 'skirt']
DRESS_KEYWORDS = ['dress', 'jumpsuit', 'romper', 'onepiece']

def _map_product_type_name_to_category(pt_name: str) -> str:
    if not isinstance(pt_name, str):
        return "top"   # safe default
    name = pt_name.lower()
    for kw in DRESS_KEYWORDS:
        if kw in name:
            return "dress"
    for kw in BOTTOM_KEYWORDS:
        if kw in name:
            return "bottom"
    for kw in TOP_KEYWORDS:
        if kw in name:
            return "top"
    # fallback: if ambiguous, return 'top' (you can change logic)
    return "top"

def predict_category(pil_image):
    """
    Accepts a PIL.Image (RGB). Returns one of: 'top', 'bottom', 'dress'.
    Uses your existing feature extractor -> nearest neighbour in dataset.
    """
    # ensure PIL image
    if not isinstance(pil_image, Image.Image):
        raise ValueError("predict_category expects a PIL.Image.Image")

    # save to a temporary file because your extract_uploaded_features expects a path
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        pil_image.save(tmp_path, format="JPEG")
        tmp.close()

        # extract features using main_code's function
        feats = extract_uploaded_features(tmp_path)   # expects a filepath
        seed_row = predict_product_type(feats)       # returns a row (dict/Series)
        product_type_name = seed_row.get('product_type_name') if hasattr(seed_row, 'get') else seed_row['product_type_name']
        category = _map_product_type_name_to_category(product_type_name)
        return category
    finally:
        # remove temporary file
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
