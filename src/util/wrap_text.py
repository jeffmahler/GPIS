import textwrap

def wrap(text, width=40):
    lines = textwrap.wrap(text, width)
    return '\n'.join(lines)