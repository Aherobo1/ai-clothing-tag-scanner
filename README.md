# ai-clothing-tag-scanner

## Overview

ai-clothing-tag-scanner is an AI-powered clothing tag identification and similarity search system. It analyzes clothing tag images, identifies brands using computer vision, and finds similar tags from a database. The system uses advanced AI techniques including image embeddings, text similarity, and (optionally) LLM-based filtering to provide accurate tag matching and recommendations.

## Features
- **Tag Identification:** Uses computer vision to identify clothing tag brands from images
- **Text-Based Matching:** Implements TF-IDF and cosine similarity for tag name matching
- **Image Similarity Search:** Uses CLIP embeddings and FAISS for visually similar tag images
- **LLM Enhancement:** Optional LLM analysis for improved similarity filtering
- **Metadata Extraction:** Provides appraisal values, years, and status information for similar tags
- **Simple Frontend:** Web UI to upload image URL, toggle LLM, and view results visually

## Tech Stack
- Python, Flask
- CLIP (Hugging Face), FAISS, scikit-learn, pandas, numpy
- OpenAI (optional, for LLM)
- HTML/CSS/JS frontend (Flask template)

## Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ai-clothing-tag-scanner
   ```
2. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv ../venv
   source ../venv/bin/activate
   ```
3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Environment Variables** (if using LLM)
   ```bash
   export OPENAI_API_KEY=your-openai-key
   ```
5. **Run the App**
   ```bash
   python app.py
   ```
   (For production, use Docker or a Linux server for stability.)

## Usage
- Go to [http://localhost:8000/](http://localhost:8000/) in your browser
- Enter a tag image URL
- Toggle "Use LLM Similarity" if desired
- Click "Scan Tag" to see results (tag info, similar images, metadata)

## File Structure
- `backend/` - Flask app, ML/DS logic, templates
- `data/` - Tag guides, expert and community CSVs
- `docs/` - Documentation (this file, API doc)

## Notes
- For best stability, run in a Linux environment or Docker.
- On Mac, the app is configured to use only one thread for all ML/numerical libraries.
- LLM similarity requires a valid OpenAI API key.

---

See `API_Documentation.md` for API details. 
