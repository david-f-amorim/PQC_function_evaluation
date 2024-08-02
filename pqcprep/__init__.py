"""
### pqcprep is a python package....

waffle waffle blah blah 

# Big test 

sss

Lorem Ipsum 
---
fdfd

## Test 2 

fff

## Test 3

fefe

"""

DIR = "pqcprep"

import os
if not os.path.isdir(os.path.join(DIR, "outputs")):
    os.mkdir(os.path.join(DIR, "outputs"))
if not os.path.isdir(os.path.join(DIR, "ampl_outputs")):
    os.mkdir(os.path.join(DIR, "ampl_outputs"))    