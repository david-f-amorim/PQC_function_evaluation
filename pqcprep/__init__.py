"""
### pqcprep is a python package....

Lorem ipsum ...

# Heading 1

Lorem ipsum ...

## Heading 2

..

## Another Heading

..

"""

DIR = "pqcprep"

import os
if not os.path.isdir(os.path.join(DIR, "outputs")):
    os.mkdir(os.path.join(DIR, "outputs"))
if not os.path.isdir(os.path.join(DIR, "ampl_outputs")):
    os.mkdir(os.path.join(DIR, "ampl_outputs"))    