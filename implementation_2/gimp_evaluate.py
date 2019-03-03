import os, sys
# get the path of this file: gimp_evaluate.py
pluginFolderPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pluginFolderPath, "src"),)

from collections import namedtuple
import numpy as np
import tensorflow as tf
from module import encoder, decoder
from utils import normalize_arr_of_imgs, denormalize_arr_of_imgs
from commons import get_img, save_img

root_models_dir = "./models"
root_models_dir = "/media/davide/Local Disk/Database/models"

# Even though we could apply style transfer to a batch of images with one pass,
# I decided to limit it to the value of 1 because the image we are processing may
# be big, and memory consumption could be a problem... Need further testing to
# evaluate the possibility to increment this value.
def inference(from_file_path, args, batch_size=1, device_t='/gpu:0'):
    imgs, style_name = args[0:2]
    # check if the specified model exists
    checkpoint_dir = os.path.join(root_models_dir, style_name, "checkpoint_long")
    assert os.path.exists(checkpoint_dir), 'Checkpoint not found!'

    ngf = 32    # args.ngf
    OPTIONS = namedtuple('OPTIONS',
                         'gf_dim  is_training')
    options = OPTIONS._make((ngf, False))

    tfconfig = tf.ConfigProto(allow_soft_placement=False) # THIS IS DIFFERENT FROM IMPL_1
    tfconfig.gpu_options.allow_growth = True
    tf.reset_default_graph()
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

        model_args = (sess, input_photo, output_photo)
        if from_file_path:
            run_from_file_paths(model_args, args)
        else:
            return run_from_layers(model_args, args[0])

def run_from_layers(model_args, imgs):
    sess, input_photo, output_photo = model_args

    processed = []
    for img in imgs:
        img = np.expand_dims(img, axis=0)
        img = sess.run(
                output_photo,
                feed_dict={
                    input_photo: normalize_arr_of_imgs(img),
                }
            )

        img = denormalize_arr_of_imgs(img[0])
        processed.append(img)

    print("Inference is finished.")

    return processed

def run_from_file_paths(model_args, input_args):
    sess, input_photo, output_photo = model_args
    imgs, style_name, output_folder, progress_update, tmp = input_args

    for i, content_path in enumerate(imgs):
        img = get_img(content_path)
        img = np.expand_dims(img, axis=0)

        img = sess.run(
                output_photo,
                feed_dict={
                    input_photo: normalize_arr_of_imgs(img),
                }
            )

        img = denormalize_arr_of_imgs(img[0])

        # generate output path
        content_name = os.path.basename(content_path)
        file_ext = os.path.splitext(content_name)
        output_path = os.path.join(output_folder, file_ext[0] + "-" + style_name + file_ext[1])

        save_img(output_path, img)
        progress_update((tmp["completed"] + float(i + 1)) / tmp["tot"])
