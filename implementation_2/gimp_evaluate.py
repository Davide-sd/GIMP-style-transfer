import os, sys
# get the path of this file: gimp_evaluate.py
pluginFolderPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pluginFolderPath, "src"),)

from collections import namedtuple
import numpy as np
import tensorflow as tf
from module import encoder, decoder
from utils import normalize_arr_of_imgs, denormalize_arr_of_imgs

root_models_dir = os.path.join(pluginFolderPath, "models/")

def inference(img, style_name, batch_size=1, device_t='/gpu:0'):
    # check if the specified model exists
    checkpoint_dir = os.path.join(root_models_dir, style_name, "checkpoint_long")
    assert os.path.exists(checkpoint_dir), 'Checkpoint not found!'

    ngf = 32    # args.ngf
    OPTIONS = namedtuple('OPTIONS',
                         'gf_dim  is_training')
    options = OPTIONS._make((ngf, False))

    tfconfig = tf.ConfigProto(allow_soft_placement=False) # THIS IS DIFFERENT FROM IMPL_1
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        # ==================== Define placeholders. ===================== #
        with tf.name_scope('placeholder'):
            input_photo = tf.placeholder(dtype=tf.float32,
                                              shape=[batch_size, None, None, 3],
                                              name='photo')

        # ===================== Wire the graph. ========================= #
        # Encode input images.
        input_photo_features = encoder(image=input_photo,
                                            options=options,
                                            reuse=False)

        # Decode obtained features.
        output_photo = decoder(features=input_photo_features,
                                    options=options,
                                    reuse=False)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print("Start inference.")

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        print(" [*] Load SUCCESS")

        img = np.expand_dims(img, axis=0)
        img = sess.run(
                output_photo,
                feed_dict={
                    input_photo: normalize_arr_of_imgs(img),
                }
            )

        img = img[0]
        img = denormalize_arr_of_imgs(img)

        print("Inference is finished.")

        return img
