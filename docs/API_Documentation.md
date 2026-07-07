# API Documentation

## POST /get_tag

Identify a clothing tag and find similar tags/images using computer vision, text similarity, and (optionally) LLM-based filtering.

### Request
- **Content-Type:** application/json
- **Body:**
  ```json
  {
    "image_url": "<image-url>",
    "use_llm": true  // Optional, default: false
  }
  ```
  - `image_url` (string, required): URL of the tag image to scan
  - `use_llm` (boolean, optional): If true, use LLM-based similarity filtering

### Response
- **Content-Type:** application/json
- **Body:**
  ```json
  {
    "results": [
      {
        "tag": "Jerzees T-Shirt Tags",
        "similar_images": [
          "https://.../tag1.jpg",
          "https://.../tag2.jpg"
        ],
        "appraisal_value": [150.0, 75.0],
        "years": ["1998", "1997"],
        "status": ["public", "private"]
      }
    ]
  }
  ```

### Example Request
```bash
curl -X POST http://localhost:8000/get_tag \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/tag_image.jpg", "use_llm": true}'
```

### Example Response
```json
{
  "results": [
    {
      "tag": "Jerzees T-Shirt Tags",
      "similar_images": [
        "https://.../tag1.jpg",
        "https://.../tag2.jpg"
      ],
      "appraisal_value": [150.0, 75.0],
      "years": ["1998", "1997"],
      "status": ["public", "private"]
    }
  ]
}
```

### Notes
- If `use_llm` is true, the system will use an LLM (OpenAI) to further filter and rank similar images.
- If no tag is identified, or no similar images are found, the response will indicate this.
- The endpoint is designed for use with the provided frontend, but can be called directly from any HTTP client.
- Requires a valid OpenAI API key if using LLM. 