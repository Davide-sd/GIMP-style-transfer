#!/usr/bin/python

# GIMP Style Transfer
# This plugin implements the Neural Style Transfer. At the moment, the following
# approaches are implemented:
#   1. implementation_1: https://github.com/lengstrom/fast-style-transfer
#   2. implementation_1: https://github.com/CompVis/adaptive-style-transfer
#   3. implementation_1: https://github.com/tensorlayer/adaptive-style-transfer

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
import sys, os
from commons_1_2 import styles, artists

pluginFolderPath = os.path.abspath(sys.argv[0])
pluginFolder = os.path.dirname(pluginFolderPath)
sys.path.insert(0, pluginFolder)

from implementation_1.main import fast_style_transfer, batch_fast_style_transfer
from implementation_2.main import artist_style_transfer, batch_artist_style_transfer
from implementation_3.main import arbitrary_style_transfer, batch_arbitrary_style_transfer

author = "Davide Sandona'"
year = "2019"
menu_filter = "<Image>/Filters/Style Transfer"
menu_batch = "<Image>/File/Batch Style Transfer"
prefix = "davide_sd_ai_"

################################################################################
############################ IMPLEMENTATION 1 ##################################
################################################################################

descr = "Apply to a target image the style of the selected painting."

register(
        prefix + "style_transfer",
        descr,
        descr,
        author,
        author,
        year,
        "Style Transfer...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_OPTION, "style", "Style", 2, tuple(styles)),
            (PF_LAYER, "layer", "Content layer", None),
        ],
        [],
        fast_style_transfer,
        menu=menu_filter)

register(
        prefix + "batch_style_transfer",
        descr,
        descr,
        author,
        author,
        year,
        "Style Transfer...",
        "",
        [
            (PF_DIRNAME, "content_folder","Content Folder",""),
            (PF_DIRNAME, "output_folder","Output Folder",""),
            (PF_OPTION, "style", "Style", 2, tuple(styles)),
        ],
        [],
        batch_fast_style_transfer,
        menu=menu_batch)

################################################################################
############################ IMPLEMENTATION 2 ##################################
################################################################################

descr = "Apply to a target image the style of a well known artist."

register(
        prefix + "artist_style_transfer",
        descr,
        descr,
        author,
        author,
        year,
        "Artist Style Transfer...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_OPTION, "style", "Style", 2, tuple(artists)),
            (PF_LAYER, "layer", "Content layer", None),
        ],
        [],
        artist_style_transfer,
        menu=menu_filter)

register(
        prefix + "batch_artist_style_transfer",
        descr,
        descr,
        author,
        author,
        year,
        "Artist Style Transfer...",
        "",
        [
            (PF_DIRNAME, "content_folder","Content Folder",""),
            (PF_DIRNAME, "output_folder","Output Folder",""),
            (PF_OPTION, "style", "Style", 2, tuple(artists)),
        ],
        [],
        batch_artist_style_transfer,
        menu=menu_batch)

################################################################################
############################ IMPLEMENTATION 3 ##################################
################################################################################

descr = "Apply to a target image the style of a specified style image."

register(
        prefix + "arbitrary_style_transfer",
        descr,
        descr,
        author,
        author,
        year,
        "Arbitrary Style Transfer...",
        "RGB*, GRAY*",
        [
            (PF_IMAGE, "image", "Input image", None),
            (PF_SLIDER, "alpha",  "Mix Factor", 100, (0, 100, 1)),
            (PF_LAYER, "layer_content", "Content layer", None),
            (PF_LAYER, "layer_style", "Style layer", None),
        ],
        [],
        arbitrary_style_transfer,
        menu=menu_filter)

descr = "Apply to a target images the style of specified style images."

register(
        prefix + "batch_arbitrary_style_transfer",
        descr,
        descr,
        author,
        author,
        year,
        "Arbitrary Style Transfer...",
        "",
        [
            (PF_DIRNAME, "content_folder","Content Folder",""),
            (PF_DIRNAME, "style_folder","Style Folder",""),
            (PF_DIRNAME, "output_folder","Output Folder",""),
            (PF_SLIDER, "alpha",  "Mix Factor", 100, (0, 100, 1)),
        ],
        [],
        batch_arbitrary_style_transfer,
        menu=menu_batch)

main()
