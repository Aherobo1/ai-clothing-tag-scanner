import pandas as pd

def aggregate_results(matched_names, expert_data, community_data, max_results=30):
    """
    Aggregate similar images, appraisal values, years, and status for matched tag names.
    Args:
        matched_names (list): List of tag names to match.
        expert_data (pd.DataFrame): Expert dataset.
        community_data (pd.DataFrame): Community dataset.
        max_results (int): Maximum number of results to return.
    Returns:
        dict: Aggregated results with images, appraisal values, years, and status.
    """
    similar_data = []
    
    for title in matched_names:
        # Handle expert data (no 'year' column)
        community_items = community_data[community_data['brand_name'] == title]
        expert_items = expert_data[expert_data['brand_name'] == title]
        
        # Process community data (has 'year' column)
        if not community_items.empty:
            # Use 'year' column if it exists, otherwise use 'year_start'
            year_col = 'year' if 'year' in community_items.columns else 'year_start'
            community_records = community_items[['front_tag', 'appraisal_value', 'key', 'status', year_col]].to_dict('records')
            # Rename year column to 'year' for consistency
            for record in community_records:
                record['year'] = record.pop(year_col) if year_col in record else None
            similar_data.extend(community_records)
        
        # Process expert data (no 'year' column)
        if not expert_items.empty:
            expert_records = expert_items[['front_tag', 'appraisal_value', 'key', 'status']].to_dict('records')
            # Add None for year since expert data doesn't have it
            for record in expert_records:
                record['year'] = None
            similar_data.extend(expert_records)
    
    # Remove duplicates by key, preserving order
    seen_keys = set()
    unique_data = []
    for item in similar_data:
        if item['key'] not in seen_keys:
            seen_keys.add(item['key'])
            unique_data.append(item)
    
    # Prepare results
    similar_images = [item['front_tag'] for item in unique_data][:max_results]
    appraisal_values = [item['appraisal_value'] for item in unique_data][:max_results]
    years = [item.get('year') for item in unique_data][:max_results]
    statuses = [item['status'] for item in unique_data][:max_results]
    
    return {
        'similar_images': similar_images,
        'appraisal_values': appraisal_values,
        'years': years,
        'statuses': statuses
    } 