
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

# --- CONFIGURATION ---
EXCEL_PATH = "dataset/training/proportional_sample_3000.xlsx"
IMAGE_FOLDER = "_images_for_python"
NUM_RECOMMENDATIONS = 4
CANDIDATES_PER_ROUND = 10

MODEL_PATH = "fashion_model.pkl"
ENCODER_PATH = "fashion_encoder.pkl"

# --- Load Data ---
df = pd.read_excel(EXCEL_PATH)
df.columns = [col.strip().lower() for col in df.columns]

# --- Gender Mapping ---
def map_gender_group(section_name):
    section = str(section_name).lower()
    female_keywords = ['womens', 'ladies', 'divided']
    male_keywords = ['mens', 'men', 'divided']
    
    if any(kw in section for kw in female_keywords):
        return "female"
    elif any(kw in section for kw in male_keywords):
        return "male"
    return None

df["gender_group"] = df["section_name"].apply(map_gender_group)
df = df[df["gender_group"].notnull()]

# --- Check for local images ---
available_images = set(os.listdir(IMAGE_FOLDER))
def image_exists(article_id):
    padded_id = str(article_id).zfill(10)
    return f"{padded_id}.jpg" in available_images

df['has_image'] = df['article_id'].apply(image_exists)
df = df[df['has_image']]
print(f"✅ {len(df)} items have local images.")

# --- Category Filters ---
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

accessory_pools = {
    'shoes': shoes,
    'jewellery': jewellery,
    'bags': bags,
    'scarves': scarves
}

# --- Helpers ---
def display_images_local(article_ids, titles):
    plt.figure(figsize=(15, 5))
    for i, (aid, title) in enumerate(zip(article_ids, titles)):
        padded_id = str(aid).zfill(10)
        image_path = os.path.join(IMAGE_FOLDER, f"{padded_id}.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                plt.subplot(1, len(article_ids), i + 1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(title)
            except Exception as e:
                print(f"❌ Error loading {image_path}: {e}")
        else:
            print(f"⚠ Image not found: {image_path}")
    plt.tight_layout()
    plt.show()

def same_style(*items):
    styles = [item['index_group_name'] for item in items if item is not None]
    return len(set(styles)) == 1

def same_gender_group(*items):
    genders = [item['gender_group'] for item in items if item is not None]
    return len(set(genders)) == 1

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
    return None, None

# --- ML Setup ---
classes = np.array([0, 1])  # 0 = dislike, 1 = like

# Load saved model and encoder if they exist
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Loaded saved model and encoder.")
else:
    model = SGDClassifier(loss='log_loss')
    encoder = OneHotEncoder(handle_unknown='ignore')
    print("⚠ No saved model found, starting fresh.")

# --- Feature extractor ---
def outfit_to_features(items):
    data = {}
    if items.get('bottom') is not None:
        data['bottom_type'] = items['bottom']['product_type_name'].lower()
        data['bottom_color'] = items['bottom']['colour_group_name'].lower()
    else:
        data['bottom_type'] = 'none'
        data['bottom_color'] = 'none'

    if items.get('top') is not None:
        data['top_type'] = items['top']['product_type_name'].lower()
        data['top_color'] = items['top']['colour_group_name'].lower()
    else:
        data['top_type'] = 'none'
        data['top_color'] = 'none'

    data['style'] = items['style'].lower()
    data['gender'] = items['gender'].lower()
    return encoder.transform(pd.DataFrame([data]))

# Initialize encoder if starting fresh
if not os.path.exists(ENCODER_PATH):
    sample_data = pd.DataFrame([{
        'bottom_type':'pant', 'bottom_color':'blue',
        'top_type':'shirt', 'top_color':'white',
        'style':'casual','gender':'male'
    }])
    encoder.fit(sample_data)

# --- Outfit generation ---
def generate_candidate():
    outfit_type = random.choice([1, 2, 3, 4])
    items = {}
    if outfit_type == 1 and not pants.empty and not shirts.empty:
        items['bottom'] = pants.sample(1).iloc[0]
        items['top'] = shirts.sample(1).iloc[0]
    elif outfit_type == 2 and not pants.empty and not tshirts.empty:
        items['bottom'] = pants.sample(1).iloc[0]
        items['top'] = tshirts.sample(1).iloc[0]
    elif outfit_type == 3 and not pants.empty and not tops.empty:
        items['bottom'] = pants.sample(1).iloc[0]
        items['top'] = tops.sample(1).iloc[0]
    elif outfit_type == 4 and not dresses.empty:
        items['bottom'] = None
        items['top'] = dresses.sample(1).iloc[0]
    else:
        return None

    items['style'] = items['top']['index_group_name'] if items['top'] is not None else items['bottom']['index_group_name']
    items['gender'] = items['top']['gender_group'] if items['top'] is not None else items['bottom']['gender_group']

    acc1, acc2 = pick_two_distinct_accessories(items['style'], items['gender'])
    if acc1 is None or acc2 is None:
        return None

    items['acc1'] = acc1
    items['acc2'] = acc2

    if not same_style(items.get('bottom'), items.get('top'), acc1, acc2) or \
       not same_gender_group(items.get('bottom'), items.get('top'), acc1, acc2):
        return None

    ids = []
    titles = []
    if items.get('bottom') is not None:
        ids.append(items['bottom']['article_id'])
        titles.append('Bottom')
    if items.get('top') is not None:
        ids.append(items['top']['article_id'])
        titles.append('Top')
    ids.extend([acc1['article_id'], acc2['article_id']])
    titles.extend(['Accessory 1', 'Accessory 2'])

    if not all(image_exists(x) for x in ids):
        return None

    items['ids'] = ids
    items['titles'] = titles
    return items

# --- ML-guided recommendations ---
count = 0
attempts = 0
max_attempts = 1000

while count < NUM_RECOMMENDATIONS and attempts < max_attempts:
    attempts += 1
    candidates = []
    for _ in range(CANDIDATES_PER_ROUND):
        candidate = generate_candidate()
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        continue

    # Rank candidates by model prediction
    best_candidate = None
    best_prob = -1
    for cand in candidates:
        try:
            X_feat = outfit_to_features(cand)
            if hasattr(model, "coef_"):
                prob = model.predict_proba(X_feat)[0,1]
            else:
                prob = 0.5
        except:
            prob = 0.5
        if prob > best_prob:
            best_prob = prob
            best_candidate = cand

    if best_candidate is None:
        continue

    print(f"🔹 Predicted probability you'll like this outfit: {best_prob:.2f}")
    display_images_local(best_candidate['ids'], best_candidate['titles'])

    # --- Get feedback ---
    feedback = input("Do you like this outfit? (1 = Yes, 0 = No): ")
    try:
        feedback = int(feedback)
        if feedback not in [0, 1]:
            feedback = 0
    except:
        feedback = 0

    # --- Update model and save ---
    try:
        X_feat = outfit_to_features(best_candidate)
        model.partial_fit(X_feat, [feedback], classes=classes)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        print("💾 Model and encoder saved.")
    except:
        pass

    count += 1

print("✅ Outfit recommendations done!")

