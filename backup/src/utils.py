import re


def color_text(texts):
    """Function to color specific original_labels for legends or axis labels."""
    for text in texts:
        if 'none' in text.get_text():
            text.set_color('blue')
        elif 'other' in text.get_text():
            text.set_color('red')


def format_metric_name(metric):
    # Convert to lowercase
    formatted_metric = metric.lower().strip()
    # Replace any sequence of one or more spaces or underscores with a single underscore
    formatted_metric = re.sub(r'[\s_]+', '_', formatted_metric)
    # Remove text inside parentheses (including parentheses)
    formatted_metric = re.sub(r'\s*\(.*?\)', '', formatted_metric)
    # Remove any leading or trailing underscores
    formatted_metric = formatted_metric.strip('_')
    return formatted_metric
