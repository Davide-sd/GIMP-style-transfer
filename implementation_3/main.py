from gimpfu import *
from commons import assert_folder_existance, createResultLayer, get_images_to_process, get_images_path, get_img, save_img
import os
from scipy.misc import imshow

from gimp_evaluate import inference

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
    pixelDataOutList = inference(False, (alpha / 100.0, content_images, style_images))

    # add the new layers to the image
    for pixelDataOut, result_name in zip(pixelDataOutList, result_names):
        createResultLayer(image, result_name, pixelDataOut)

    # Close the undo group.
    pdb.gimp_image_undo_group_end(image)
    # End progress.
    pdb.gimp_progress_end()

def batch_arbitrary_style_transfer(content_folder, style_folder, output_folder, alpha):
    assert_folder_existance(content_folder, "content")
    assert_folder_existance(style_folder, "style")

    if output_folder == None:
        output_folder = os.path.join(content_folder, "output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    content_images_path = get_images_path(content_folder)
    style_images_path = get_images_path(style_folder)

    def progress_update(percentage):
        pdb.gimp_progress_update(percentage)

    inference(True, (alpha / 100.0, content_images_path, style_images_path, output_folder, progress_update))

    # End progress.
    pdb.gimp_progress_end()
