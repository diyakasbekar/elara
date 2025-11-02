from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import io
from PIL import Image
from classifier import predict_category  # classifier for uploaded image
from main_recommend import (
    generate_recommendation_for_uploaded,
    generate_recommendation_for_dataset
)

# ✅ Initialize Flask app
app = Flask(__name__)

# Folder containing your dataset images
IMAGE_FOLDER = os.path.join(os.getcwd(), "_images_for_python")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# ✅ Home route
@app.route('/')
def index():
    return render_template('index.html')


# ✅ Helper to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ✅ Serve dataset images safely
@app.route('/dataset_image/<path:filename>')
def dataset_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


# ✅ List all dataset items (for “Select Outfits” / “Browse Dataset”)
@app.route('/get_dataset_items')
def get_dataset_items():
    items = []
    for file in os.listdir(IMAGE_FOLDER):
        if allowed_file(file):
            art_id = os.path.splitext(file)[0]
            items.append({
                'article_id': art_id,
                'image_url': f'/dataset_image/{file}'
            })
    return jsonify({'items': items})


# ✅ Handle dataset item selection → get recommendations
@app.route('/recommend_dataset_item/<article_id>')
def recommend_dataset_item(article_id):
    recs = generate_recommendation_for_dataset(article_id)
    return jsonify({'recommendations': recs})


# ✅ Handle user-uploaded image → classify → recommend
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read uploaded image directly into memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Step 1: Predict product category (top/bottom/dress)
        category = predict_category(image)
        print(f"🧥 Detected category: {category}")

        # Step 2: Save temp image so recommendation function can read it
        temp_path = os.path.join(os.getcwd(), "temp_upload.jpg")
        image.save(temp_path)

        # Step 3: Generate recommendations
        recs = generate_recommendation_for_uploaded(temp_path, upload_type=category)

        # Step 4: Clean up temporary image
        if os.path.exists(temp_path):
            os.remove(temp_path)

        print("✅ Recommendations generated successfully.")
        return jsonify({'category': category, 'recommendations': recs})

    except Exception as e:
        import traceback
        traceback.print_exc()  # 🔍 Full error shown in console
        print("❌ Error processing image:", str(e))
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)
