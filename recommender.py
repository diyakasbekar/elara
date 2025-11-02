# recommender.py
import pandas as pd
import random

df = pd.read_excel('dataset/training/proportional_sample_3000.xlsx')

def get_color_group(color_name):
    # Simple way to group similar shades
    color_map = {
        'Blue': ['Navy', 'Cyan', 'Sky', 'Teal'],
        'Red': ['Maroon', 'Pink', 'Crimson'],
        'Green': ['Olive', 'Mint', 'Lime'],
        'Black': ['Black', 'Gray'],
        'White': ['White', 'Cream', 'Beige']
    }
    for main, subs in color_map.items():
        if any(sub.lower() in color_name.lower() for sub in subs):
            return main
    return 'Neutral'

def recommend_items(category, base_color):
    main_color = get_color_group(base_color)

    if category == 'topwear':
        bottoms = df[df['product_group_name'].str.contains('Bottom', case=False, na=False)]
        accessories = df[df['product_group_name'].str.contains('Accessory|Footwear|Jewellery', case=False, na=False)]
        bottom_choice = bottoms.sample(2)
        acc_choice = accessories.sample(2)
        return bottom_choice, acc_choice

    elif category == 'bottomwear':
        tops = df[df['product_group_name'].str.contains('Top', case=False, na=False)]
        accessories = df[df['product_group_name'].str.contains('Accessory|Footwear|Jewellery', case=False, na=False)]
        top_choice = tops.sample(2)
        acc_choice = accessories.sample(2)
        return top_choice, acc_choice

    elif category == 'fullbody':
        accessories = df[df['product_group_name'].str.contains('Accessory|Footwear|Jewellery', case=False, na=False)]
        acc_choice = accessories.sample(4)
        return None, acc_choice
