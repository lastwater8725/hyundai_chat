import os
import json
from glob import glob
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LAParamers 

input_dir = 