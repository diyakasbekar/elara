# main_recommend.py
"""
Unified recommendation API used by app.py.

Functions exposed:
- generate_recommendation_for_uploaded(upload_input, upload_type=None)
    upload_input can be:
      * a filesystem path to an uploaded image (string) OR
      * a category string "top"/"bottom"/"dress"

- generate_recommendation_for_dataset(article_id)
    Generate recommendations for a dataset item (user clicked an item).
"""

import os
import random

from main_code import (
    extract_uploaded_features,
    predict_product_type,
    generate_outfit,
    get_row_by_article_id,
    df as dataset_df
)

IMAGE_FOLDER = "_images_for_python"

# map categories to the upload_type values used by generate_outfit
CATEGORY_TO_UPLOAD_TYPE = {
    'top': 'top',
    'bottom': 'bottom',
    'pant': 'bottom',
    'dress': 'dress'
}

def _pick_random_seed_row_for_category(category):
    """
    If we only have a category (top/bottom/dress) — choose a random row from dataset
    that matches. This gives generate_outfit a realistic seed_row when the uploaded
    image was only classified by category.
    """
    cat = category.lower()
    if cat == 'top':
        qry = dataset_df['product_type_name'].str.contains('shirt|t-shirt|top|blouse|tee', case=False, na=False)
    elif cat == 'bottom' or cat == 'pant':
        qry = dataset_df['product_type_name'].str.contains('pant|jean|trouser|short|skirt', case=False, na=False)
    elif cat == 'dress':
        qry = dataset_df['product_type_name'].str.contains('dress|jumpsuit|romper|onepiece', case=False, na=False)
    else:
        qry = dataset_df['product_type_name'].notnull()

    subset = dataset_df[qry]
    if subset.empty:
        # fallback: any random row
        return dataset_df.sample(1).iloc[0]
    return subset.sample(1).iloc[0]

def generate_recommendation_for_uploaded(upload_input, upload_type=None):
    """
    upload_input: either a path to an uploaded image (string) or a category string
                  returned by classifier: "top" / "bottom" / "dress".
    upload_type: optional override ('top', 'bottom', 'dress') to pass to generator.
    returns: dict {"recommendations": [ {article_id, title, image_url}, ... ] }
    """
    try:
        # Case A: upload_input is a path to an image file
        if isinstance(upload_input, str) and os.path.exists(upload_input):
            feats = extract_uploaded_features(upload_input)
            seed_row = predict_product_type(feats)
            upload_type_to_use = upload_type or CATEGORY_TO_UPLOAD_TYPE.get(
                seed_row.get('product_type_name', '').lower().split()[0], 'top'
            )
            outfit = generate_outfit(seed_row, uploaded=True, upload_type=upload_type_to_use)

        # Case B: upload_input is a category string (e.g. "top", "bottom", "dress")
        elif isinstance(upload_input, str):
            category = upload_input.lower()
            seed_row = _pick_random_seed_row_for_category(category)
            upload_type_to_use = upload_type or CATEGORY_TO_UPLOAD_TYPE.get(category, category)
            # call generate_outfit with uploaded=True so we follow the rule-based combos
            outfit = generate_outfit(seed_row, uploaded=True, upload_type=upload_type_to_use)

        else:
            # unsupported type
            return {"recommendations": []}

        if not outfit or 'ids' not in outfit:
            return {"recommendations": []}

        recommendations = []
        for aid, title in zip(outfit['ids'], outfit['titles']):
            padded_id = str(aid).zfill(10)
            image_url = f"/dataset_image/{padded_id}.jpg"
            recommendations.append({
                "article_id": padded_id,
                "title": title,
                "image_url": image_url
            })

        # success
        print("✅ Recommendations generated successfully.")
        return {"recommendations": recommendations}

    except Exception as e:
        print("❌ Error in generate_recommendation_for_uploaded:", e)
        return {"recommendations": []}


def generate_recommendation_for_dataset(article_id):
    """
    article_id: string or int representing dataset item
    returns: dict {"recommendations": [...]}
    """
    try:
        row = get_row_by_article_id(article_id)
        if row is None:
            return {"recommendations": []}

        outfit = generate_outfit(row, uploaded=False)
        if not outfit or "ids" not in outfit:
            return {"recommendations": []}

        recommendations = []
        for aid, title in zip(outfit["ids"], outfit["titles"]):
            padded_id = str(aid).zfill(10)
            image_url = f"/dataset_image/{padded_id}.jpg"
            recommendations.append({
                "article_id": padded_id,
                "title": title,
                "image_url": image_url
            })

        return {"recommendations": recommendations}

    except Exception as e:
        print("❌ Error in generate_recommendation_for_dataset:", e)
        return {"recommendations": []}
