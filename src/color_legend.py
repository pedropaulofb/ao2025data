def color_text(texts):
    """Function to color specific texts for legends or axis labels."""
    for text in texts:
        if 'none' in text.get_text():
            text.set_color('blue')
        elif 'other' in text.get_text():
            text.set_color('red')
