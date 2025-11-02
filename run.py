import pandas as pd
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

EXCEL_PATH = "dataset/training/proportional_sample_3000.xlsx"
IMAGE_FOLDER = "_images_for_python"

# Load Data
df = pd.read_excel(EXCEL_PATH)
df.columns = [col.strip().lower() for col in df.columns]

# Keep only rows with local images
available_images = set(os.listdir(IMAGE_FOLDER))
df['has_image'] = df['article_id'].apply(lambda x: f"{str(x).zfill(10)}.jpg" in available_images)
df = df[df['has_image']]

# --- Feature extractor ---
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_feat = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model_feat.predict(x)

# Compute features
dataset_features = []
valid_rows = []

for _, row in df.iterrows():
    art_id = str(row['article_id']).zfill(10)
    local_path = os.path.join(IMAGE_FOLDER, f"{art_id}.jpg")
    try:
        feat = extract_features(local_path)
        dataset_features.append(feat)
        valid_rows.append(row)
    except Exception as e:
        print(f"Error processing {local_path}: {e}")

dataset_features = np.vstack(dataset_features)

# Save features for future use
np.save("dataset_features.npy", dataset_features)
np.save("valid_rows.npy", np.array(valid_rows, dtype=object))
print("✅ Precomputed features saved.")
def predict_product_type(upload_img_path, top_n=NUM_TOP_MATCHES):
    user_feat = extract_features(upload_img_path)
    sims = cosine_similarity(user_feat, dataset_features)[0]
    top_idx = sims.argsort()[-top_n:][::-1]
    
    # Ensure valid_rows is a list of dicts
    rows_list = [dict(row) if isinstance(row, pd.Series) else row for row in valid_rows]

    top_types = [rows_list[i]['product_type_name'].lower() for i in top_idx]
    counts = {}
    for t in top_types:
        counts[t] = counts.get(t, 0) + 1
    predicted_type = max(counts, key=counts.get)
    for i in top_idx:
        if rows_list[i]['product_type_name'].lower() == predicted_type:
            return rows_list[i]
    return rows_list[top_idx[0]]
