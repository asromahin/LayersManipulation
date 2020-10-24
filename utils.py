def only_chars(text):
    return ''.join([t for t in text if not t.isdigit()])