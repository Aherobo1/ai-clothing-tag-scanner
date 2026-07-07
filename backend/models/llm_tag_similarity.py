import openai
import json
import base64
from PIL import Image
import io
import requests
from typing import List, Dict
import os

class LLMTagSimilarity:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize LLM-based tag similarity analyzer
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: Model to use for analysis
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def encode_image_to_base64(self, image_url: str) -> str:
        """Convert image URL to base64 for API"""
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return img_str
        except Exception as e:
            print(f"Error encoding image {image_url}: {e}")
            return None

    def analyze_tag_similarity(self, query_image_url: str, candidate_images: List[str], max_candidates: int = 30) -> List[Dict]:
        """
        Use LLM to analyze similarity between query tag and candidate tags
        Args:
            query_image_url: URL of the query tag image
            candidate_images: List of candidate tag image URLs
            max_candidates: Maximum number of candidates to analyze
        Returns:
            List of candidates with similarity scores and explanations
        """
        candidates = candidate_images[:max_candidates]
        query_base64 = self.encode_image_to_base64(query_image_url)
        if not query_base64:
            return []
        candidate_base64s = []
        for img_url in candidates:
            base64_img = self.encode_image_to_base64(img_url)
            if base64_img:
                candidate_base64s.append(base64_img)
        prompt = """
        You are an expert in t-shirt tag authentication. Your job is to strictly assess visual authenticity between a query tag and several candidate tags.
        For each candidate, return:
        1. similarity_score (0-100): Based on **visual design fidelity** (not just text). How visually similar the tag looks. not just a color combination but color is very important. for example, a tag could have a whilte background a blue text and a red stripe is not similar at all to the same kind of tag with same white background and same text style and everything but has a red text with blue stripe. Also text colors are very important, two tags can look alike but text color are different... they are not similar. color is simple... for example, sky blue, navy blue is all blue... red, wine, is all the same. then focus on the overall style of the tag.. could be the same brand or name but diffrent style isnt similar. background color and text color is very important. so for example, if a tag background is black, similar tags are tags that are visiually similar and have black background as well. if a tag background is white, similar tags are tags that are visiually similar and have white background and so on.
        Output a JSON array of objects like:
        {
            "candidate_index": 0,
            "similarity_score": 42,
        }
        """
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{query_base64}",
                    "detail": "high"
                }
            }
        ]
        for i, base64_img in enumerate(candidate_base64s):
            content.append({"type": "text", "text": f"Candidate {i+1}:"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "high"
                }
            })
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=2000,
                temperature=0.1
            )
            analysis_text = response.choices[0].message.content
            try:
                start_idx = analysis_text.find('[')
                end_idx = analysis_text.rfind(']') + 1
                json_str = analysis_text[start_idx:end_idx]
                results = json.loads(json_str)
                for i, result in enumerate(results):
                    result['original_url'] = candidates[i]
                return results
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Response: {analysis_text}")
                return []
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return []

    def filter_similar_tags(self, query_image_url: str, candidate_images: List[str], similarity_threshold: float = 70.0) -> List[Dict]:
        """
        Filter candidates based on LLM similarity analysis
        Args:
            query_image_url: URL of the query tag image
            candidate_images: List of candidate tag image URLs
            similarity_threshold: Minimum similarity score to include
        Returns:
            Filtered list of similar tags with scores
        """
        analysis_results = self.analyze_tag_similarity(query_image_url, candidate_images)
        filtered_results = [
            result for result in analysis_results 
            if result.get('similarity_score', 0) >= similarity_threshold
        ]
        filtered_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        return filtered_results 