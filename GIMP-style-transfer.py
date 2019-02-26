#!/usr/bin/python

# GIMP Style Transfer
# This plugin implements the Neural Style Transfer. At the moment, the following
# approaches are implemented:
#   1. https://github.com/lengstrom/fast-style-transfer
#   2. https://github.com/CompVis/adaptive-style-transfer
#   3. https://github.com/tensorlayer/adaptive-style-transfer
#
# Copyright (c) 2019 Davide Sandona'
# sandona [dot] davide [at] gmail [dot] com
# https://github.com/Davide-sd/GIMP-style-transfer

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from gimpfu import *
import os
import sys
import numpy as np

from implementation_1.gimp_evaluate import inference as inference_1
# TODO: WHY THE RESULTING LAYER IS A LITTLE BIT BIGGER THAN INPUT LAYER WITH implementation_2???
from implementation_2.gimp_evaluate import inference as inference_2
from implementation_3.gimp_evaluate import inference as inference_3

# the following three functions are used by all the plugins
def channelData(layer):
    """ Returns a numpy array of the size [height, width, bpp] of the input layer.
    bpp stands for Bytes Per Pixel.
    """
    w, h = layer.width, layer.height
    region = layer.get_pixel_rgn(0, 0, w, h)
    pixChars = region[:, :]
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(h, w, bpp)

def createResultLayer(image, name, result):
    """ Create and add a new layer to the image.
    Input parameters:
        image   : the image where to add the new layer
        name    : the layer name
        result  : the pixels color informations for the new layer
    """
    rlBytes = np.uint8(result).tobytes()
    rl = gimp.Layer(image, name, result.shape[1], result.shape[0],
                  image.active_layer.type, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()

def get_images_to_process(layer):
    """ Check if the given layer is a Layer Group.
    Input parameters:
        layer: the layer to check

    Output parameters:
        images  :   a list of numpy arrays obtained with channelData,
                    representing the selected layers
        names   :   a list of names of selected layers
    """
    images = []
    names = []
    if (type(layer) == gimp.GroupLayer):
        for l in layer.layers:
            images.append(channelData(l))
            names.append(l.name)
    else:
        images.append(channelData(layer))
        names.append(layer.name)

    return images, names

################################################################################
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
        if os.path.exists(checkpoint_path):
            availableModels.append(k)
    return availableModels

# extract the value from the correct dictionary
def getModelName(modelsDict, availableModels, selected_index):
    return modelsDict[availableModels[selected_index]]

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

def fast_style_transfer(image, style, layer):
    if style == -1:
        pdb.gimp_message("There are no model available: exiting execution.")
        return

    model_name = getModelName(style_transfer_dict, styles, style)
    style_transfer_function(image, layer, model_name, inference_1)

def artist_style_transfer(image, style, layer):
    if style == -1:
        pdb.gimp_message("There are no model available: exiting execution.")
        return

    model_name = getModelName(artist_style_transfer_dict, artists, style)
    style_transfer_function(image, layer, model_name, inference_2)

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
    pixelDataOutList = style_transfer(content_images, style_name)

    # add the new layers to the image
    for pixelDataOut, result_name in zip(pixelDataOutList, result_names):
        createResultLayer(image, result_name, pixelDataOut)

    # Close the undo group.
    pdb.gimp_image_undo_group_end(image)
    # End progress.
    pdb.gimp_progress_end()

register(
        "python_fu_style_transfer",
        "Apply to a target image the style of the selected picture.",
        "Apply to a target image the style of the selected picture.",
        "Davide Sandona'",
        "Davide Sandona'",
        "2019",
        "Style Transfer...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_OPTION, "style", "Style", 2, tuple(styles)),
            (PF_LAYER, "layer", "Content layer", None),
        ],
        [],
        fast_style_transfer,
        menu="<Image>/Filters/Style Transfer")

register(
        "python_fu_artist_style_transfer",
        "Apply to a target image the style of a well known artist.",
        "Apply to a target image the style of a well known artist.",
        "Davide Sandona'",
        "Davide Sandona'",
        "2019",
        "Artist Style Transfer...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_OPTION, "style", "Style", 2, tuple(artists)),
            (PF_LAYER, "layer", "Content layer", None),
        ],
        [],
        artist_style_transfer,
        menu="<Image>/Filters/Style Transfer")

################################################################################
############################ IMPLEMENTATION 3 ##################################
################################################################################

def arbitrary_style_transfer(image, alpha, layer_content, layer_style):
    # Set up an undo group, so the operation will be undone in one step.
    pdb.gimp_image_undo_group_start(image)

    # get content and style images and names
    content_images, content_names = get_images_to_process(layer_content)
    style_images, style_names = get_images_to_process(layer_style)

    # get layers name
    result_names = []
    for c_name in content_names:
        for s_name in style_names:
            result_names.append(c_name + "-" + s_name)

    # apply the style transfer
    pixelDataOutList = inference_3(content_images, style_images, alpha / 100.0)

    # add the new layers to the image
    for pixelDataOut, result_name in zip(pixelDataOutList, result_names):
        createResultLayer(image, result_name, pixelDataOut)

    # Close the undo group.
    pdb.gimp_image_undo_group_end(image)
    # End progress.
    pdb.gimp_progress_end()

register(
        "python_fu_arbitrary_style_transfer",
        "Apply to a target image the style of a specified style image.",
        "Apply to a target image the style of a specified style image.",
        "Davide Sandona'",
        "Davide Sandona'",
        "2019",
        "Arbitrary Style Transfer...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_SLIDER, "alpha",  "Alpha", 100, (0, 100, 1)),
            (PF_LAYER, "layer_content", "Content layer", None),
            (PF_LAYER, "layer_style", "Style layer", None),
        ],
        [],
        arbitrary_style_transfer,
        menu="<Image>/Filters/Style Transfer")

main()
