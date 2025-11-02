# main_code.py (enhanced rules + color + formality matching)
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import random

IMAGE_FOLDER = "_images_for_python"
FEATURES_FILE = "dataset_features.npy"
VALID_ROWS_FILE = "valid_rows.npy"

# Load dataset features and valid rows
dataset_features = np.load(FEATURES_FILE)
valid_rows_raw = list(np.load(VALID_ROWS_FILE, allow_pickle=True))
valid_rows = [r.to_dict() if isinstance(r, pd.Series) else dict(r) for r in valid_rows_raw]
df = pd.DataFrame(valid_rows)

# Load model for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_feat = Model(inputs=base_model.input, outputs=base_model.output)

# --- Category Filters ---
def filter_by_keywords(df_local, keywords):
    return df_local[df_local['product_type_name'].str.contains('|'.join(keywords), case=False, na=False)]

pants = filter_by_keywords(df, ['pant', 'jean', 'trouser'])
tops = filter_by_keywords(df, ['top', 'shirt', 't-shirt', 'blouse'])
dresses = filter_by_keywords(df, ['dress', 'jumpsuit'])
shoes = filter_by_keywords(df, ['shoe', 'sandal', 'heel', 'boot', 'footwear'])
jewellery = filter_by_keywords(df, ['jewellery', 'jewelry', 'necklace', 'ring', 'bracelet', 'earring'])
scarves = filter_by_keywords(df, ['scarf', 'shawl'])

# --- Helpers ---
def extract_uploaded_features(img_path):
    """Extract deep features from uploaded image."""
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model_feat.predict(x)
    return feat

def predict_product_type(upload_features, top_n=5):
    """Find similar dataset items and guess product type."""
    sims = cosine_similarity(upload_features, dataset_features)[0]
    top_idx = sims.argsort()[-top_n:][::-1]
    top_types = [valid_rows[i]['product_type_name'].lower() for i in top_idx]
    most_common = max(set(top_types), key=top_types.count)
    for i in top_idx:
        if valid_rows[i]['product_type_name'].lower() == most_common:
            return valid_rows[i]
    return valid_rows[top_idx[0]]

def get_row_by_article_id(article_id):
    s = str(article_id).zfill(10)
    match = df[df['article_id'].astype(str).str.zfill(10) == s]
    return match.iloc[0] if not match.empty else None

# --- Color & Formality Helpers ---
def detect_color_family(color_name):
    color_name = str(color_name).lower()
    if any(c in color_name for c in ["black", "grey", "gray", "charcoal"]):
        return "neutral_dark"
    if any(c in color_name for c in ["white", "cream", "beige", "ivory"]):
        return "neutral_light"
    if any(c in color_name for c in ["blue", "navy", "teal"]):
        return "cool"
    if any(c in color_name for c in ["red", "pink", "orange", "yellow"]):
        return "warm"
    if any(c in color_name for c in ["green", "olive", "khaki"]):
        return "earthy"
    return "misc"

def detect_formality(name):
    name = str(name).lower()
    if any(k in name for k in ["suit", "blazer", "formal", "office", "shirt", "trouser", "heel"]):
        return "formal"
    if any(k in name for k in ["jean", "t-shirt", "casual", "sneaker", "hoodie", "jogger"]):
        return "casual"
    return "neutral"

def match_color_and_formality(pool, seed_color, seed_formality, gender_group):
    if pool.empty:
        return pool
    pool = pool[pool['gender_group'] == gender_group]
    if 'colour_group_name' in pool.columns:
        pool['color_family'] = pool['colour_group_name'].apply(detect_color_family)
    else:
        pool['color_family'] = 'misc'
    pool['formality'] = pool['product_type_name'].apply(detect_formality)

    # filter by color family and formality
    matched = pool[
        (pool['color_family'] == seed_color) | 
        (pool['formality'] == seed_formality)
    ]
    return matched if not matched.empty else pool

# --- Rule-Based Outfit Generator ---
def generate_outfit(seed_row, uploaded=True, upload_type=None):
    if seed_row is None:
        return None

    seed = seed_row.to_dict() if isinstance(seed_row, pd.Series) else seed_row
    gender_group = seed.get('gender_group', None)
    style_name = seed.get('index_group_name', None)
    product_name = seed.get('product_type_name', '').lower()

    # detect seed color + formality
    seed_color = detect_color_family(seed.get('colour_group_name', ''))
    seed_formality = detect_formality(product_name)

    items = []

    def pick(pool, count=2):
        p = match_color_and_formality(pool, seed_color, seed_formality, gender_group)
        if p.empty:
            return []
        return p.sample(min(count, len(p))).to_dict('records')

    # === Rules ===
    if upload_type == "top":
        bottoms = pick(pants, 2)
        footwear = pick(shoes, 1)
        acc = pick(jewellery, 1) if random.random() > 0.5 else pick(scarves, 1)
        items = bottoms + footwear + acc

    elif upload_type == "bottom":
        topset = pick(tops, 2)
        footwear = pick(shoes, 1)
        acc = pick(jewellery, 1) if random.random() > 0.5 else pick(scarves, 1)
        items = topset + footwear + acc

    elif upload_type == "dress":
        footwear = pick(shoes, 2)
        acc1 = pick(jewellery, 1)
        acc2 = pick(scarves, 1)
        items = footwear + acc1 + acc2

    else:
        # fallback
        topset = pick(tops, 1)
        bottoms = pick(pants, 1)
        items = topset + bottoms

    if not items:
        return None

    ids = [str(int(it['article_id'])).zfill(10) for it in items]
    titles = [it.get('product_type_name', 'Item') for it in items]
    return {"ids": ids, "titles": titles}
