from __future__ import print_function
import sys, os
# get the path of this file: gimp_evaluate.py
pluginFolderPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pluginFolderPath, "src"),)
import transform, numpy as np, pdb
import tensorflow as tf

root_models_dir = "./models"

def inference(img, style_name, batch_size=1, device_t='/gpu:0'):
    # check if the specified model exists
    checkpoint_dir = os.path.join(pluginFolderPath, "models/", style_name + ".ckpt")
    assert os.path.exists(checkpoint_dir), 'Checkpoint not found!'

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with tf.Session(config=soft_config) as sess:
        input_img_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=(batch_size,) + img.shape,
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

        img = np.expand_dims(img, axis=0)
        img = sess.run(
                preds,
                feed_dict={
                    input_img_placeholder: img,
                }
            )

        img = img[0]

        return img
