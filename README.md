# GIMP Style Transfer

**This is a prototype plugin (or a proof of concept, if you wish) intended to demonstrate the feasibility to expand GIMP abilities with state of the art machine learning algorithms.**

This plugin adds the ability to perform Neural Style Transfer: this techniques reconstruct a *content image* (the selected GIMP layer) with the style of a *style image* (selected with the plugin).

You may have used apps that do this kind of processing, like [Prisma](https://prisma-ai.com/) (or something similar).

At the moment, the following approaches are implemented in the plugin:
* [Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer): transfer the style of  well know paintings to your images. This approach produces really nice results. As of now, only 6 style models are available.  
I have not yet received authorization to release its code. Follow *Install instruction* below to get this implementation working.
* [Adaptive Style Transfer implementation](https://github.com/CompVis/adaptive-style-transfer): apply the style of a given artist to your image. This is a very interesting approach, but it is also much more difficult to get appreciable results. Feel free to experiment with it, also changing the blend mode and opacity of the resulting layer to get a satisfying final effect.  
As of now, 13 artists have been implemented. From here on, I'll refer to this approach as **Artist Style Transfer** because it better represent what it actual does.

Please, take a look at the repositories linked above to get a visual idea of what this plugin allows us to do.

## Requirements

I developed this plugin on Ubuntu 18.04, GIMP 2.10.8, Python 2.7. AFAIK *tensorflow* requires Python 3.5+ on Windows to work, therefore I strongly believe that this plugin will not work on Windows.

* Install [Tensorflow](https://www.tensorflow.org/install) on the Python environment used by GIMP (usually version 2.7) with `pip install tensorflow`.  
I tested this plugin with `tensorflow` CPU only. Feel free to test it also with `tensorflow-gpu` by installing the module `pip install tensorflow-gpu` (you need a CUDA-enabled GPU card).  
Note: if you are using an *old* CPU that does not support AVX instruction set, you will have to either install Tensorflow version 1.5 (`pip install tensorflow tensorflow==1.5`) or [build the latest version from source](https://www.tensorflow.org/install/source) (it may take a while on older CPU).

* You also need to install the following Python packages: `PIL, numpy, scipy`

## Install

1. Open the terminal, move to the GIMP plug-ins folder (usually `~/.config/GIMP/2.10/plug-ins`).

2. `git clone https://github.com/Davide-sd/GIMP-style-transfer.git`

3. `cd GIMP-style-transfer`

4. Change permission to allow execution: `chmod +x GIMP-style-transfer.py`

5. Don't forget to apply the following steps: *Setting up Fast Style Transfer* and/or *Setting up Artist Style Transfer*.

### Setting up [Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer)

Since I have not yet received any reply to my inquire regarding the possibility of releasing that code, you have to follow these steps to get it working.

1. Download the repository [Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer) and extract it wherewever you want.
2. Copy the file `fast-style-transfer/src/transform.py` into the plugin folder `GIMP-style-transfer/implementation_1/src`.
3. Downloads the models for this implementation [located at this link](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ) (this will results in 115MB of uncompressed data). Extract the `.ckpt` files into the folder `GIMP-style-transfer/implementation_1/models`.  
You can download only the models you are interested in (the plugin will adapt accordingly).

At this point you should be able to use the plugin located in the menu `Filters/Style Transfer/Style Transfer...`.

### Setting up [Artist Style Transfer](https://github.com/CompVis/adaptive-style-transfer)

1. Downloads the models for the this implementation [located at this link](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/XXVKT5grAquXNqi).  **Warning: this will results in 9.1GB of uncompressed data!!!!**  
Extract the archives into the folder `GIMP-style-transfer/implementation_2/models`.  
You can download only the models you are interested in (the plugin will adapt accordingly).

At this point you should be able to use the plugin located in the menu `Filters/Style Transfer/Artist Style Transfer...`.

## Usage

1. Open an image (start playing with small images, read the next section carefully!).
2. In the layer palette, select the layer you want to apply the style transfer.
3. Click on `Filters/Style Transfer/Artist Style Transfer...` or `Filters/Style Transfer/Style Transfer...`
4. Select the desired option from the combobox.
5. Click `Ok` and wait for the process to complete.

## Memory Usage

Style Transfer requires a *huge* amount of RAM. I would advise against testing this plugin on systems with less than 4GB of RAM.

To give you an example, with a `1200px X 800px` image I noticed a boost in RAM usage of 2.5GB (when executing the algorithms on CPU only).

The memory usage increases rapidly with the input image size! Start testing with smaller images, then gradually increase the image size.

**It would be nice if someone (with a decent amount of RAM) could create a plot relating image size (pixels) vs memory used (see section *TODO* point 1).**

## TODO

1. Compute an estimate of the memory necessary to perform the computation on the input image. Do not perform the computation if this value exceeds the total amount of free memory (and inform the user).

2. Look at the possibility to extract only the meaningfull weights from the [Artist Style Transfer implementation](https://github.com/CompVis/adaptive-style-transfer) models, in order to significantly reduce their file sizes.

3. Artist Style Transfer produces an output image slightly bigger then input. Why and how to fix it?

4. Look for strategies to get this plugin working on Windows.

## FAQ

### Who is this plugin for?

This plugin is meant for GIMP developers. The objectives of this plugin are:

1. To create curiosity between existing and new GIMP developers towards machine learning algorithms.
2. Proof of concept that GIMP can be further and successfully extended by using machine learning algorithms.
3. Identify the best strategies to get GIMP and Tensorflow (or any other machine learning frameworks) to work togheter (even on Windows).
4. Experiment with different implementations, understanding their pros and cons, understand why a given model produces a certain result.


### Who is this plugin NOT for?

This plugin is NOT meant for the generic GIMP user. Don't even try to go through the hassle of setting up your system for this plugin. You probably would end up disappointed with the results...


### I would like more models to play with. What should I do?

Look at the respective implementations pages: you will find the instructions to train new models. This procedure is only recomended if you use a decent CUDA-enabled GPU card. If you are going to train new models, feel free to share them on this repository.
