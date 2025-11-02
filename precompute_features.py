import pandas as pd
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

EXCEL_PATH = "dataset/training/proportional_sample_3000.xlsx"
IMAGE_FOLDER = "_images_for_python"

# Load data
df = pd.read_excel(EXCEL_PATH)
df.columns = [col.strip().lower() for col in df.columns]

# Gender mapping
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

# Only items with local images
available_images = set(os.listdir(IMAGE_FOLDER))
df['has_image'] = df['article_id'].apply(lambda x: f"{str(x).zfill(10)}.jpg" in available_images)
df = df[df['has_image']]

# Load MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_feat = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model_feat.predict(x)

# Precompute features and store corresponding rows
dataset_features = []
valid_rows = []

for _, row in df.iterrows():
    art_id = str(row['article_id']).zfill(10)
    local_path = os.path.join(IMAGE_FOLDER, f"{art_id}.jpg")
    try:
        feat = extract_features(local_path)
        dataset_features.append(feat)
        valid_rows.append(row.to_dict())  # Save as dict for easy reloading
    except Exception as e:
        print(f"Error processing {local_path}: {e}")

dataset_features = np.vstack(dataset_features)

# Save both features and rows
np.save("dataset_features.npy", dataset_features)
np.save("valid_rows.npy", valid_rows)  # <-- now saved!

print(f"✅ Features and valid rows saved for {len(valid_rows)} items.")
