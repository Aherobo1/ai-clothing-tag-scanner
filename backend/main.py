from flask import Flask, request, jsonify, render_template
import os
from PIL import Image
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
try:
    torch.set_num_threads(1)
except Exception as e:
    print("Could not set torch num threads:", e)
import json
import numpy as np
from services.image_utils import download_image
from models.tag_identification import TagIdentification
from models.tag_match import get_best_match
from data.data_utils import load_tag_guides, load_expert_data, load_community_data
from services.image_similarity import load_index, search_similar_images, transform_image
from services.result_aggregation import aggregate_results
from models.llm_tag_similarity import LLMTagSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'front_tag_embeddings.index')
TAG_GUIDES_PATH = os.path.join(DATA_DIR, 'tag_guides_clean.json')
EXPERT_DATA_PATH = os.path.join(DATA_DIR, 'expert_data.csv')
COMMUNITY_DATA_PATH = os.path.join(DATA_DIR, 'community_data.csv')

# Global variables for lazy loading
_index = None
_tag_identifier = None
_llm_analyzer = None

def get_index():
    global _index
    if _index is None:
        _index = load_index(INDEX_PATH)
    return _index

def get_tag_identifier():
    global _tag_identifier
    if _tag_identifier is None:
        _tag_identifier = TagIdentification(endpoint_id="22mdrm9fckjera")
    return _tag_identifier

def get_llm_analyzer():
    global _llm_analyzer
    if _llm_analyzer is None:
        _llm_analyzer = LLMTagSimilarity()
    return _llm_analyzer

def get_temp_image_path():
    return os.path.join(DATA_DIR, 'downloaded_image.jpg')

def get_score_front_tag_simple(indices_expert_front_tag, expert_data):
    """
    Simplified version of get_score_front_tag that works with our data structure
    """
    # For now, just return the first N images from expert_data
    # In a full implementation, you'd map indices to actual image URLs
    similar_images = expert_data['front_tag'].dropna().tolist()[:30]
    appraisal_values = expert_data['appraisal_value'].dropna().tolist()[:30]
    keys = expert_data['key'].dropna().tolist()[:30]
    statuses = expert_data['status'].dropna().tolist()[:30]
    
    return {
        "results": [
            {
                "similar images": {
                    "front_tag": similar_images
                },
                "appraisal_value": appraisal_values,
                "keys": keys,
                "predictions": [1.0] * len(similar_images)  # Default prediction
            }
        ]
    }

def process_images_list_batch(images_list, query_image):
    """
    Process historical images from tag_guides and return sorted scores
    """
    # Simplified version - in full implementation, you'd compute actual similarity scores
    if not images_list:
        return []
    
    # For now, just return the first image with a default score
    return [(images_list[0].get('year', 'Unknown'), 1.0)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Tag scan app is running!'})

@app.route('/get_tag', methods=['POST'])
def get_tag():
    try:
        data = request.json
        image_url = data.get('image_url')
        use_llm = data.get('use_llm', False)
        if not image_url:
            return jsonify({'error': 'No image_url provided'}), 400

        # Download image
        img_path = get_temp_image_path()
        download_image(image_url, img_path)
        query_image = Image.open(img_path)

        # Tag identification
        tag_identifier = get_tag_identifier()
        tag = tag_identifier.identify_tag(image_url)
        text = tag['response'] if tag and 'response' in tag else None
        if not text:
            return jsonify({'error': 'No tag identified'}), 404

        # Load tag guides
        loaded_data = load_tag_guides(TAG_GUIDES_PATH)

        # Text matching
        best_match = get_best_match(text, tag_guides_path=TAG_GUIDES_PATH, top_n=1)
        if not best_match or best_match[0]['similarity_score'] == 0.0:
            return jsonify({'message': 'Invalid Tag', 'similar images': []}), 404
        extracted_text = best_match[0]['matched_data']['name']
        print("Extracted Tag: ", extracted_text)

        # Load data
        expert_data = load_expert_data(EXPERT_DATA_PATH)
        community_data = load_community_data(COMMUNITY_DATA_PATH)

        # Image similarity search
        index = get_index()
        distances_expert_front_tag, indices_expert_front_tag = search_similar_images(
            query_image, index, top_k=30)

        # Get predictions and scores (simplified version)
        result_dict = get_score_front_tag_simple(indices_expert_front_tag, expert_data)
        initial_similar_images = result_dict['results'][0]['similar images']['front_tag']

        # Efficient text processing (as in main app)
        community_titles = set(community_data['brand_name'].dropna())
        expert_titles = set(expert_data['brand_name'].dropna())
        all_titles = list(community_titles.union(expert_titles))
        
        print(f"Total titles found: {len(all_titles)}")
        print(f"Sample titles: {all_titles[:5]}")

        # Vectorize texts efficiently
        tag_name = best_match[0]['matched_data']['name']
        print(f"Looking for matches to: '{tag_name}'")
        
        # Extract the main brand name (e.g., "Jerzees" from "Jerzees T-Shirt Tags")
        main_brand = tag_name.split()[0] if tag_name else ""
        print(f"Main brand: '{main_brand}'")
        
        # Ensure we have some titles to compare against
        if len(all_titles) == 0:
            print("No titles found in data, using fallback")
            similar_images = expert_data['front_tag'].dropna().tolist()[:30]
            appraisal_values = expert_data['appraisal_value'].dropna().tolist()[:30]
            statuses = expert_data['status'].dropna().tolist()[:30]
            years = ["Unknown"]
            
            response = {
                'results': [
                    {
                        'tag': extracted_text,
                        'similar_images': similar_images,
                        'appraisal_value': appraisal_values,
                        'years': years,
                        'status': statuses
                    }
                ]
            }
            print("Final response (fallback):", response)
            return jsonify(response)
        
        # Try to find exact or partial matches first
        exact_matches = [title for title in all_titles if main_brand.lower() in title.lower()]
        print(f"Exact matches found: {len(exact_matches)}")
        if exact_matches:
            print(f"Sample exact matches: {exact_matches[:3]}")
        
        # If we have exact matches, use them
        if exact_matches:
            top_titles = [(title, 1.0) for title in exact_matches[:10]]
        else:
            # Fall back to TF-IDF similarity
            try:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([tag_name] + all_titles)
                similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
                print("Similarities: ", similarities)
                print(f"Max similarity: {np.max(similarities)}")
                print(f"Min similarity: {np.min(similarities)}")

                # Get top similar titles with a lower threshold
                top_indices = np.argsort(similarities)[-10:][::-1]  # Get top 10 instead of 5
                top_titles = [(all_titles[i], similarities[i]) for i in top_indices if similarities[i] >= 0.1]  # Lower threshold
                
                print(f"Top titles found: {top_titles}")
                
                # If no titles meet the threshold, use the top 5 anyway
                if not top_titles:
                    print("No titles meet threshold, using top 5 anyway")
                    top_titles = [(all_titles[i], similarities[i]) for i in top_indices[:5]]
            except Exception as e:
                print(f"TF-IDF processing failed: {e}")
                # Use fallback - just take some random titles
                top_titles = [(title, 0.5) for title in all_titles[:10]]

        # Process similar images efficiently (as in main app)
        try:
            similar_data = []
            for title, score in top_titles:
                community_items = community_data[community_data['brand_name'] == title]
                expert_items = expert_data[expert_data['brand_name'] == title]

                for items in [community_items, expert_items]:
                    if not items.empty:
                        similar_data.extend(items[['front_tag', 'appraisal_value', 'key', 'status']].to_dict('records'))

            # Remove duplicates while preserving order
            seen_keys = set()
            unique_data = []
            for item in similar_data:
                if item['key'] not in seen_keys:
                    seen_keys.add(item['key'])
                    unique_data.append(item)

            # Prepare results
            similar_images = [item['front_tag'] for item in unique_data]
            similar_images = similar_images[:30]
            print("Similar Images: ", similar_images)
            appraisal_values = [item['appraisal_value'] for item in unique_data]
            keys = [item['key'] for item in unique_data]
            statuses = [item['status'] for item in unique_data]
            
        except Exception as e:
            print(f"Data processing failed: {e}")
            # Fallback to using expert data directly
            similar_images = expert_data['front_tag'].dropna().tolist()[:30]
            appraisal_values = expert_data['appraisal_value'].dropna().tolist()[:30]
            statuses = expert_data['status'].dropna().tolist()[:30]
            keys = expert_data['key'].dropna().tolist()[:30]

        # Process historical images (as in main app)
        images_list = []
        for tag in loaded_data["tag_guides"]:
            if extracted_text == tag["name"]:
                print("Got it")
                print("Extracted Text : ", extracted_text)
                images_list = tag["images"]
                break

        print("Images List: ", images_list)
        try:
            sorted_scores = process_images_list_batch(images_list, query_image)
            print("Sorted Scores: ", sorted_scores)
            years = [year for year, _ in sorted_scores]
            print("Extracted years:", years)
        except Exception as e:
            print("Error during process_images_list_batch or years extraction:", e)
            import traceback; traceback.print_exc()
            sorted_scores = []
            years = []
        # Continue with the rest of the logic even if this fails

        if extracted_text == "Fruit of the Loom ":
            years = ["1970"]
            response = {"message": 'No result found'}
            print("Final response (no result):", response)
            return jsonify(response)

        # LLM similarity filter (only if use_llm is True)
        if use_llm:
            llm_analyzer = get_llm_analyzer()
            llm_results = llm_analyzer.filter_similar_tags(image_url, similar_images, similarity_threshold=75.0)
            similar_images_final = [item.get('original_url') for item in llm_results if item.get('original_url')]
            if not similar_images_final:
                similar_images_final = similar_images
        else:
            similar_images_final = similar_images

        # Filter other fields to match LLM-selected images
        def filter_by_images(images_final, images_all, *fields_all):
            image_to_index = {img: idx for idx, img in enumerate(images_all)}
            filtered_fields = []
            for field in fields_all:
                filtered = [field[image_to_index[img]] for img in images_final if img in image_to_index]
                filtered_fields.append(filtered)
            return filtered_fields

        appraisal_values_final, keys_final, statuses_final = filter_by_images(
            similar_images_final, similar_images, appraisal_values, keys, statuses
        )

        # Prepare response (matching main app format)
        response = {
            'results': [
                {
                    'tag': extracted_text,
                    'similar_images': similar_images_final,
                    'appraisal_value': appraisal_values_final,
                    'status': statuses_final
                }
            ]
        }
        print("Final response:", response)
        return jsonify(response)
    except Exception as e:
        print("Fatal error in /get_tag:", e)
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 