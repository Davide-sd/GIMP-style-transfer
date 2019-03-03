from gimpfu import *
import sys, os
import numpy as np
import scipy.misc
from PIL import Image
from collections import defaultdict

from commons import *

################################################################################
###################### Variables and methods used in ###########################
######################### IMPLEMENTATION 1 AND 2 ###############################
################################################################################

# The user may not have downloaded all the models. Need to check which models are
# available to properly add the options on the plugin comboboxes.
def checkModelsAvailability(modelsDict, implementation):
    availableModels = []
    pluginFolderPath = os.path.dirname(os.path.abspath(__file__))

    for k, v in modelsDict.items():
        checkpoint_path = os.path.join(pluginFolderPath, implementation, "models/")
        if implementation == "implementation_1":
            checkpoint_path = os.path.join(checkpoint_path, v + ".ckpt")
        elif implementation == "implementation_2":
            checkpoint_path = os.path.join(checkpoint_path, v, "checkpoint_long")
            checkpoint_path = os.path.join("/media/davide/Local Disk/Database/models", v, "checkpoint_long")
        if os.path.exists(checkpoint_path):
            availableModels.append(k)
    return availableModels

# In the following dictionaries:
# the keys are used in the plugin UI combobox
# the values indicate the model name or directory
# Note: if you change the values, the edits should be reflected in the folders structure as well
style_transfer_dict = {
    "The Scream": "the-scream",
    "La Muse": "la-muse",
    "Udnie": "udnie",
    "Wave": "great-wave",
    "Rain Princess": "rain-princess",
    "The Shipwreck of the minotaur": "the-shipwreck"
}

artist_style_transfer_dict = {
    "Berthe Morisot": "model_morisot",
    "Claude Monet": "model_monet",
    "Edvard Munch": "model_munch",
    "El Greco": "model_el-greco",
    "Ernst Ludwig Kirchner": "model_kirchner",
    "Jackson Pollock": "model_pollock",
    "Nicholas Roerich": "model_roerich",
    "Pablo Picasso": "model_picasso",
    "Paul Cezanne": "model_cezanne",
    "Paul Gauguin": "model_gauguin",
    "Samuel Peploe": "model_peploe",
    "Vincent Van Gogh": "model_van-gogh",
    "Wassily Kandisky": "model_kandinsky"
}

# get a list of the available models
styles = checkModelsAvailability(style_transfer_dict, "implementation_1")
artists = checkModelsAvailability(artist_style_transfer_dict, "implementation_2")

# Sort alphabetically to better search the menu options in the plugin UI.
# This is useful to get the correct value when the user launch the execution,
# because (AFAIK) PF_OPTION only returns the selected index
styles.sort()
artists.sort()

# extract the value from the correct dictionary
def getModelName(modelsDict, availableModels, selected_index):
    return modelsDict[availableModels[selected_index]]

def style_transfer_function(image, layer, style_name, style_transfer):
    # Set up an undo group, so the operation will be undone in one step.
    pdb.gimp_image_undo_group_start(image)

    # get content images data and names
    content_images, content_names = get_images_to_process(layer)

    # prepare the output layers name
    result_names = []
    for c_name in content_names:
        result_names.append(c_name + "-" + style_name)

    # apply the style transfer
    pixelDataOutList = style_transfer(False, (content_images, style_name))

    # add the new layers to the image
    for pixelDataOut, result_name in zip(pixelDataOutList, result_names):
        createResultLayer(image, result_name, pixelDataOut)

    # Close the undo group.
    pdb.gimp_image_undo_group_end(image)
    # End progress.
    pdb.gimp_progress_end()

def batch_style_transfer_function(content_folder, output_folder, style_name, style_transfer):
    assert_folder_existance(content_folder, "content")

    if output_folder == None:
        output_folder = os.path.join(content_folder, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    content_images_path = get_images_path(content_folder)

    group_imgs_by_shape = defaultdict(list)
    for img_path in content_images_path:
        img_shape = Image.open(img_path).size
        group_imgs_by_shape[img_shape].append(img_path)

    def progress_update(percentage):
        pdb.gimp_progress_update(percentage)

    # this variable is used to compute the progress inside gimp_evaluate
    tmp = {
        "completed": 0,
        "tot": len(content_images_path)
    }

    for img_shape in group_imgs_by_shape:
        style_transfer(True,
            (group_imgs_by_shape[img_shape],
            style_name,
            output_folder,
            progress_update,
            tmp))
        tmp["completed"] += len(group_imgs_by_shape[img_shape])

    # End progress.
    pdb.gimp_progress_end()
