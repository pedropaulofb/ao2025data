
def append_unique_preserving_order(existing_list, new_keys):
    """Append keys to the list, preserving the original order and ensuring no duplicates."""
    for key in new_keys:
        if key not in existing_list:
            existing_list.append(key)
