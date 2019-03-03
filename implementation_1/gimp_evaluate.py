from __future__ import print_function
import sys, os
# get the path of this file: gimp_evaluate.py
pluginFolderPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pluginFolderPath, "src"),)
import transform, numpy as np, pdb
import tensorflow as tf
from commons import get_img, save_img

root_models_dir = "./models"
batch_size = 1

# Even though we could apply style transfer to a batch of images with one pass,
# I decided to limit it to the value of 1 because the image we are processing may 
# be big, and memory consumption could be a problem... Need further testing to
# evaluate the possibility to increment this value.
def inference(from_file_path, args, batch_size=1, device_t='/gpu:0'):
    imgs, style_name = args[0:2]

    # check if the specified model exists
    checkpoint_dir = os.path.join(pluginFolderPath, "models/", style_name + ".ckpt")
    assert os.path.exists(checkpoint_dir), 'Checkpoint not found!'

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    tf.reset_default_graph()
    with tf.Session(config=soft_config) as sess:
        img_shape = None
        if from_file_path:
            img_shape = get_img(imgs[0]).shape
        else:
            img_shape = imgs[0].shape
        input_img_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=(batch_size,) + img_shape,
                name='input_img_placeholder')

        preds = transform.net(input_img_placeholder)

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        model_args = (sess, input_img_placeholder, preds)
        if from_file_path:
            run_from_file_paths(model_args, args)
        else:
            return run_from_layers(model_args, args[0])

def run_from_layers(model_args, imgs):
    sess, input_img_placeholder, preds = model_args

    processed = []
    for img in imgs:
        img = np.expand_dims(img, axis=0)
        img = sess.run(
                preds,
                feed_dict={
                    input_img_placeholder: img,
                }
            )

        img = np.clip(img[0], 0, 255).astype(np.uint8)
        processed.append(img)

    return processed

def run_from_file_paths(model_args, input_args):
    sess, input_img_placeholder, preds = model_args
    imgs, style_name, output_folder, progress_update, tmp = input_args

    for i, content_path in enumerate(imgs):
        img = get_img(content_path)
        img = np.expand_dims(img, axis=0)

        img = sess.run(
                preds,
                feed_dict={
                    input_img_placeholder: img,
                }
            )

        img = np.clip(img[0], 0, 255).astype(np.uint8)

        # generate output path
        content_name = os.path.basename(content_path)
        file_ext = os.path.splitext(content_name)
        output_path = os.path.join(output_folder, file_ext[0] + "-" + style_name + file_ext[1])

        save_img(output_path, img)
        progress_update((tmp["completed"] + float(i + 1)) / tmp["tot"])
