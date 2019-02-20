# Fast Style Transfer

This implementation is based on [Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer).

Since I have not yet received any reply to my inquire regarding the possibility of releasing that code, you have to follow these steps to get it working.

1. Download the repository [Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer) and extract it wherewever you want.
2. Copy the file `fast-style-transfer/src/transform.py` into the plugin folder `GIMP-style-transfer/implementation_1/src`.
3. Downloads the models for this implementation [located at this link](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ) (this will results in 115MB of uncompressed data). Extract the `.ckpt` files into the folder `GIMP-style-transfer/implementation_1/models`.  
You can download only the models you are interested in (the plugin will adapt accordingly).

At this point you should be able to use the plugin located in the menu `Filters/Style Transfer/Style Transfer...`.
