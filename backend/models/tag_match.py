import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def get_best_match(tag_response, tag_guides_path=None, top_n=1):
    """
    Find the best match for a tag_response using cosine similarity on local tag_guides_clean.json.

    Args:
        tag_response (str): The input tag to be matched.
        tag_guides_path (str): Path to the local tag_guides_clean.json file.
        top_n (int): Number of top matches to return (default is 1).

    Returns:
        list: A list of top_n best matches with similarity scores and matched data.
    """
    if tag_guides_path is None:
        tag_guides_path = os.path.join(os.path.dirname(__file__), '../data/tag_guides_clean.json')
    with open(tag_guides_path, 'r') as f:
        data = json.load(f)
    names = [item['name'] for item in data['tag_guides']]
    vectorizer = TfidfVectorizer().fit(names + [tag_response])
    name_vectors = vectorizer.transform(names)
    response_vector = vectorizer.transform([tag_response])
    cosine_similarities = cosine_similarity(response_vector, name_vectors).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    best_matches = []
    for index in top_indices:
        best_matches.append({
            'matched_name': data['tag_guides'][index]['name'],
            'similarity_score': round(cosine_similarities[index], 4),
            'matched_data': data['tag_guides'][index]
        })
    return best_matches 