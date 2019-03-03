from gimpfu import *
from commons_1_2 import styles, style_transfer_dict, getModelName, style_transfer_function, batch_style_transfer_function

from gimp_evaluate import inference

def fast_style_transfer(image, style, layer):
    if style == -1:
        pdb.gimp_message("There are no model available: exiting execution.")
        return

    model_name = getModelName(style_transfer_dict, styles, style)
    style_transfer_function(image, layer, model_name, inference)

def batch_fast_style_transfer(content_folder, output_folder, style):
    if style == -1:
        pdb.gimp_message("There are no model available: exiting execution.")
        return

    model_name = getModelName(style_transfer_dict, styles, style)
    batch_style_transfer_function(content_folder, output_folder, model_name, inference)
