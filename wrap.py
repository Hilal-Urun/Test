from ansiwrap import wrap

from colors import *


def wrap_paragraph(msg, width=70):
    if width < 1:
        print(msg)
    else:
        print("\n".join(wrap(msg, width)))


def wrap_print(msg, width=70):
    for paragraph in msg.split("\n\n"):
        for line in paragraph.split("\n"):
            wrap_paragraph(line, width)
        # print("")
