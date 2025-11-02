import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# --- CONFIGURATION ---
IMAGE_FOLDER = "_images_for_python"
NUM_TOP_MATCHES = 5
FEATURES_FILE = "dataset_features.npy"  # Precomputed features
VALID_ROWS_FILE = "valid_rows.npy"      # Precomputed valid rows

# --- Load precomputed features ---
print("🔄 Loading precomputed features...")
dataset_features = np.load(FEATURES_FILE)
valid_rows = list(np.load(VALID_ROWS_FILE, allow_pickle=True))
valid_rows = [pd.Series(r) if isinstance(r, dict) else r for r in valid_rows]
df = pd.DataFrame(valid_rows)  # Use valid_rows directly
print(f"✅ Loaded features for {len(valid_rows)} items.")

# --- Category Pools ---
def filter_by_keywords(df, keywords):
    return df[df['product_type_name'].str.contains('|'.join(keywords), case=False, na=False)]

pants = filter_by_keywords(df, ['pant', 'jean', 'trouser'])
shirts = filter_by_keywords(df, ['shirt'])
tshirts = filter_by_keywords(df, ['t-shirt'])
tops = filter_by_keywords(df, ['top'])
dresses = filter_by_keywords(df, ['dress'])
shoes = filter_by_keywords(df, ['shoe'])
jewellery = filter_by_keywords(df, ['jewellery', 'jewelry', 'necklace', 'ring', 'bracelet', 'earring'])
bags = filter_by_keywords(df, ['bag'])
scarves = filter_by_keywords(df, ['scarf'])
accessory_pools = {'shoes': shoes, 'jewellery': jewellery, 'bags': bags, 'scarves': scarves}

print("✅ Pools ready. Outfit generation is now instant.")

# --- Helpers ---
def display_images_local(article_ids, titles, uploaded_img_path=None):
    total = len(article_ids) + (1 if uploaded_img_path else 0)
    plt.figure(figsize=(4 * total, 5))
    idx = 1
    if uploaded_img_path:
        try:
            img = Image.open(uploaded_img_path)
            plt.subplot(1, total, idx)
            plt.imshow(img)
            plt.axis('off')
            plt.title("Your Upload")
            idx += 1
        except Exception as e:
            print(f"⚠ Could not display uploaded image: {e}")

    for aid, title in zip(article_ids, titles):
        padded_id = str(aid).zfill(10)
        image_path = os.path.join(IMAGE_FOLDER, f"{padded_id}.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                plt.subplot(1, total, idx)
                plt.imshow(img)
                plt.axis('off')
                plt.title(title)
                idx += 1
            except Exception as e:
                print(f"❌ Error loading {image_path}: {e}")
    plt.tight_layout()
    plt.show()

def pick_two_distinct_accessories(style, gender_group):
    categories = list(accessory_pools.keys())
    random.shuffle(categories)
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            pool1 = accessory_pools[cat1]
            pool2 = accessory_pools[cat2]
            pool1_filtered = pool1[(pool1['index_group_name'] == style) & (pool1['gender_group'] == gender_group)]
            pool2_filtered = pool2[(pool2['index_group_name'] == style) & (pool2['gender_group'] == gender_group)]
            if pool1_filtered.empty or pool2_filtered.empty:
                continue
            acc1 = pool1_filtered.sample(1).iloc[0]
            acc2 = pool2_filtered.sample(1).iloc[0]
            if acc1['article_id'] != acc2['article_id']:
                return acc1, acc2
    # fallback if none found
    all_acc = pd.concat(accessory_pools.values())
    all_acc = all_acc[all_acc['gender_group'] == gender_group]
    if len(all_acc) >= 2:
        sample_acc = all_acc.sample(2)
        return sample_acc.iloc[0], sample_acc.iloc[1]
    return None, None

# --- Feature extractor for uploaded image ---
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_feat = Model(inputs=base_model.input, outputs=base_model.output)

def extract_uploaded_features(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model_feat.predict(x)

# --- Predict product type ---
def predict_product_type(upload_features, top_n=NUM_TOP_MATCHES):
    sims = cosine_similarity(upload_features, dataset_features)[0]
    top_idx = sims.argsort()[-top_n:][::-1]
    top_types = [valid_rows[i]['product_type_name'].lower() for i in top_idx]
    counts = {}
    for t in top_types:
        counts[t] = counts.get(t, 0) + 1
    predicted_type = max(counts, key=counts.get)
    for i in top_idx:
        row = valid_rows[i]
        if row['product_type_name'].lower() == predicted_type:
            return row
    return valid_rows[top_idx[0]]

# --- Outfit generation ---
def generate_outfit(seed_row, uploaded=True, upload_type=None):
    items = {}
    product_name = seed_row['product_type_name'].lower()
    gender_group = seed_row['gender_group']
    style_name = seed_row['index_group_name']

    if uploaded:
        if upload_type == "top":
            pool_bottom = pants[(pants['gender_group']==gender_group) & (pants['index_group_name']==style_name)]
            if pool_bottom.empty: pool_bottom = pants[pants['gender_group']==gender_group]
            if pool_bottom.empty: return None
            items['bottom'] = pool_bottom.sample(1).iloc[0]
        elif upload_type == "bottom":
            pool_top = pd.concat([shirts, tshirts, tops])
            pool_top = pool_top[(pool_top['gender_group']==gender_group) & (pool_top['index_group_name']==style_name)]
            if pool_top.empty: pool_top = pd.concat([shirts, tshirts, tops])[ pd.concat([shirts, tshirts, tops])['gender_group']==gender_group]
            if pool_top.empty: return None
            items['top'] = pool_top.sample(1).iloc[0]
        elif upload_type == "dress":
            items['top'] = None
            items['bottom'] = None
        else:
            return None
    else:
        pool_top = pd.concat([shirts, tshirts, tops])
        pool_top = pool_top[pool_top['gender_group']==gender_group]
        pool_bottom = pants[pants['gender_group']==gender_group]
        if pool_top.empty or pool_bottom.empty: return None
        items['top'] = pool_top.sample(1).iloc[0]
        items['bottom'] = pool_bottom.sample(1).iloc[0]

    acc1, acc2 = pick_two_distinct_accessories(style_name, gender_group)
    if acc1 is None or acc2 is None: return None
    items['acc1'], items['acc2'] = acc1, acc2

    ids, titles = [], []
    if items.get('bottom') is not None:
        ids.append(items['bottom']['article_id']); titles.append('Bottom')
    if items.get('top') is not None:
        ids.append(items['top']['article_id']); titles.append('Top')
    ids.extend([acc1['article_id'], acc2['article_id']])
    titles.extend(['Accessory 1', 'Accessory 2'])
    items['ids'], items['titles'] = ids, titles
    return items

# --- MAIN ---
print("\n👗 Welcome! Upload your clothing image for outfit recommendation.")

upload_type = ""
while upload_type not in ["top", "bottom", "dress", "skip"]:
    upload_type = input("Select the type of item you are uploading (top / bottom / dress) or type 'skip' to random: ").strip().lower()

img_path = ""
if upload_type != "skip":
    img_path = input("Enter full path of your clothing image: ").strip('"').strip()

if img_path and os.path.exists(img_path):
    print("🔍 Analyzing your uploaded image...")
    upload_features = extract_uploaded_features(img_path)
    seed_row = predict_product_type(upload_features)
    outfit = generate_outfit(seed_row, uploaded=True, upload_type=upload_type)
    if outfit:
        display_images_local(outfit['ids'], outfit['titles'], uploaded_img_path=img_path)
    else:
        print("⚠ Could not generate outfit for your uploaded image.")
else:
    print("➡ Skipping upload. Generating a random outfit instead...")
    random_row = df.sample(1).iloc[0]
    outfit = generate_outfit(random_row, uploaded=False)
    if outfit:
        display_images_local(outfit['ids'], outfit['titles'])
    else:
        print("⚠ Could not generate a random outfit.")
