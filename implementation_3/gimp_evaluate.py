import os, sys
import numpy as np
import tensorflow as tf
from commons import get_img, save_img

# get the path of this file: gimp_evaluate.py
pluginFolderPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(pluginFolderPath, "src"),)
from models import Decoder, Encoder

ENCODER_PATH = 'models/pretrained_vgg19_encoder_model.npz'
DECODER_PATH = 'models/pretrained_vgg19_decoder_model.npz'
ENCODER_PATH = os.path.join(pluginFolderPath, ENCODER_PATH)
DECODER_PATH = os.path.join(pluginFolderPath, DECODER_PATH)

# Normalizes the `content_features` with scaling and offset from `style_features`.
def AdaIN(content_features, style_features, alpha=1, epsilon=1e-5):

    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keep_dims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keep_dims=True)

    normalized_content_features = tf.nn.batch_normalization(
        content_features, content_mean, content_variance, style_mean, tf.sqrt(style_variance), epsilon
    )
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features

def inference(from_file_path, args):
    with tf.Graph().as_default(), tf.Session() as sess:
        alpha = args[0]

        encoder = Encoder()
        decoder = Decoder()

        content_input = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content_input')
        style_input = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='style_input')

        # switch RGB to BGR
        content = tf.reverse(content_input, axis=[-1])
        style = tf.reverse(style_input, axis=[-1])
        # preprocess image
        content = encoder.preprocess(content)
        style = encoder.preprocess(style)

        # encode image
        # we should initial global variables before restore model
        enc_c_net = encoder.encode(content, 'content/')
        enc_s_net = encoder.encode(style, 'style/')

        # pass the encoded images to AdaIN
        target_features = AdaIN(enc_c_net.outputs, enc_s_net.outputs, alpha=alpha)

        # decode target features back to image
        dec_net = decoder.decode(target_features, prefix="decoder/")

        generated_img = dec_net.outputs

        # deprocess image
        generated_img = encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        sess.run(tf.global_variables_initializer())

        encoder.restore_model(sess, ENCODER_PATH, enc_c_net)
        encoder.restore_model(sess, ENCODER_PATH, enc_s_net)
        decoder.restore_model(sess, DECODER_PATH, dec_net)

        model_args = (sess, generated_img, content_input, style_input)
        if from_file_path:
            run_from_file_paths(model_args, args)
        else:
            return run_from_layers(model_args, args)

def run_from_layers(model_args, input_args):
    sess, generated_img, content_input, style_input = model_args
    alpha, content_images, style_images = input_args

    processed = []
    for content_image in content_images:
        for style_image in style_images:
            content_tensor = np.expand_dims(content_image, axis=0)
            style_tensor = np.expand_dims(style_image, axis=0)
            result = sess.run(
                        generated_img,
                        feed_dict={
                            content_input: content_tensor,
                            style_input: style_tensor
                        })
            processed.append(result[0])

    return processed

def run_from_file_paths(model_args, input_args):
    sess, generated_img, content_input, style_input = model_args
    alpha, content_images, style_images, output_folder, progress_update = input_args

    n_imgs = len(style_images) * len(content_images)
    for i, content_image in enumerate(content_images):
        content_image = get_img(content_image)
        for j, style_image in enumerate(style_images):
            style_image = get_img(style_image)

            content_tensor = np.expand_dims(content_image, axis=0)
            style_tensor = np.expand_dims(style_image, axis=0)
            result = sess.run(
                        generated_img,
                        feed_dict={
                            content_input: content_tensor,
                            style_input: style_tensor
                        })
            # assemble the output file name
            style_img_split = os.path.splitext(os.path.basename(style_images[j]))
            content_img_split = os.path.splitext(os.path.basename(content_images[i]))
            output_image_path = content_img_split[0] + "-" + style_img_split[0] + content_img_split[1]
            output_image_path = os.path.join(output_folder, output_image_path)
            save_img(output_image_path, result[0])
            # update the gimp progress bar
            curr_img = float(i * len(style_images) + (j + 1))
            progress_update(curr_img / n_imgs)
