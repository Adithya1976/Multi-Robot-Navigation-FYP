def tuple2string(t):
    if isinstance(t, str):
        return t
    return ' '.join(str(item) + ' ' for item in t)
