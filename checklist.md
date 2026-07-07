# Refactoring Checklist: Backend Folder Structure

## Goal
Restructure the backend code into a more modular, maintainable format.

## Steps

- [ ] 1. Create new folders inside `backend/`: `api/`, `data/`, `models/`, `services/`, `utils/`
- [ ] 2. Move `app.py` to `main.py` in `backend/`
- [ ] 3. Move `data_utils.py` to `backend/data/data_utils.py`
- [ ] 4. Move `image_similarity.py` and `image_utils.py` to `backend/services/`
- [ ] 5. Move `llm_tag_similarity.py`, `tag_identification.py`, and `tag_match.py` to `backend/models/`
- [ ] 6. Move `result_aggregation.py` to `backend/services/`
- [ ] 7. Move `front_tag_embeddings.index` and `tag_identification_cache.json` to `backend/data/`
- [ ] 8. Update all import statements to reflect the new structure
- [ ] 9. Test the application to ensure everything works
- [ ] 10. Update documentation if needed 