from gimpfu import *
import sys, os
import numpy as np
import scipy.misc

################################################################################
################# Methods used in all implementations ##########################
################################################################################

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

    # TODO: need to find a better way to deal with these cases
    def get_rgb_pixelData(layer):
        pixelData = channelData(layer)
        # if monochrome image
        if len(pixelData.shape) == 2:
            pixelData = np.stack((pixelData, pixelData, pixelData), axis=2)
        # if alpha channel is present (R,G,B,Alpha)
        elif len(pixelData.shape) > 2 and pixelData.shape[2] > 3:
            pixelData = pixelData[:, :, 0:3]
        return pixelData

    if (type(layer) == gimp.GroupLayer):
        for l in layer.layers:
            images.append(get_rgb_pixelData(l))
            names.append(l.name)
    else:
        images.append(get_rgb_pixelData(layer))
        names.append(layer.name)

    return images, names

def get_img(src):
   img = scipy.misc.imread(src, mode='RGB')
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img, img, img))
   return img

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def get_images_path(folder):
    # load file paths from the folder
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    # get pictures to process
    return [f for f in files if file_is_image(f)]

def file_is_image(file_name):
    allowed_import_types = ["jpg","jpeg","tiff","tif","bmp","png"]
    # get the extension
    ext = os.path.splitext(file_name)[1]
    # rip off the . from the extension
    ext = ext.replace(".", "")
    if ext.lower() in allowed_import_types:
        return True
    return False

def assert_folder_existance(folder, name):
    assert folder != None, "No " + name + " folder provided."
    assert os.path.exists(folder), "The " + name + " folder does not exist."
