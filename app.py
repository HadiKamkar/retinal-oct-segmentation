# app.py
from flask import Flask, jsonify, request, render_template, send_file, make_response
from flask_cors import CORS
import numpy as np
import os
import scipy.io
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, init, nd, autograd
import cv2
import base64
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
CORS(app)
run_with_ngrok(app)
#################################
from keras.models import load_model, model_from_json
import tensorflow as tf




num_workers = mx.context.num_gpus()
if num_workers:
  context = mx.gpu(0)
else:
  context = mx.cpu(0)

class network(gluon.nn.HybridBlock):
    def convoluting_part(self, input_channels, output_channels, kernel_size=3):
        shrink_net = gluon.nn.HybridSequential()
        with shrink_net.name_scope():
            shrink_net.add(
                gluon.nn.Conv2D(in_channels=input_channels, channels=output_channels, kernel_size=kernel_size,
                                activation='relu'))
            shrink_net.add(gluon.nn.BatchNorm(in_channels=output_channels))
            shrink_net.add(
                gluon.nn.Conv2D(in_channels=output_channels, channels=output_channels, kernel_size=kernel_size,
                                activation='relu'))
            shrink_net.add(gluon.nn.BatchNorm(in_channels=output_channels))
            shrink_net.add(gluon.nn.MaxPool2D(pool_size=(2, 2)))
        return shrink_net

    def deconvoluting_part(self, input_channels, hidden_channel, output_channels, kernel_size=3):
        expand_net = gluon.nn.HybridSequential()
        with expand_net.name_scope():
            expand_net.add(gluon.nn.Conv2D(channels=hidden_channel, kernel_size=kernel_size, activation='relu'))
            expand_net.add(gluon.nn.BatchNorm())
            expand_net.add(gluon.nn.Conv2D(channels=hidden_channel, kernel_size=kernel_size, activation='relu'))
            expand_net.add(gluon.nn.BatchNorm())
            expand_net.add(gluon.nn.Conv2DTranspose(channels=output_channels, kernel_size=kernel_size, strides=(2, 2),
                                                    padding=(1, 1), output_padding=(1, 1)))
        return expand_net

    def plateau_block(self, input_channels, output_channels):
        plateau_net = gluon.nn.HybridSequential()
        with plateau_net.name_scope():
            plateau_net.add(gluon.nn.Conv2D(channels=512, kernel_size=3, activation='relu'))
            plateau_net.add(gluon.nn.BatchNorm())
            plateau_net.add(gluon.nn.Conv2D(channels=512, kernel_size=3, activation='relu'))
            plateau_net.add(gluon.nn.BatchNorm())
            plateau_net.add(gluon.nn.Conv2DTranspose(channels=256, kernel_size=3, strides=(2, 2), padding=(1, 1),
                                                     output_padding=(1, 1)))
        return plateau_net

    def output_block(self, input_channels, hidden_channel, output_channels, kernel_size=3):
        x = gluon.nn.HybridSequential()
        with x.name_scope():
            x.add(gluon.nn.Conv2D(in_channels=input_channels, channels=hidden_channel, kernel_size=kernel_size,
                                  activation='relu'))
            x.add(gluon.nn.BatchNorm(in_channels=hidden_channel))
            x.add(gluon.nn.Conv2D(in_channels=hidden_channel, channels=hidden_channel, kernel_size=kernel_size,
                                  activation='relu'))
            x.add(gluon.nn.BatchNorm(in_channels=hidden_channel))
            x.add(gluon.nn.Conv2D(in_channels=hidden_channel, channels=output_channels, kernel_size=kernel_size,
                                  padding=(1, 1), activation='relu'))
            x.add(gluon.nn.BatchNorm(in_channels=output_channels))
        return x

    def concatenate(self, upsampling_block, conv_block):
        padding = upsampling_block.shape[2] - conv_block.shape[2]
        mid_padding = padding // 2
        padded_conv_block = mx.nd.pad(conv_block, mode="edge",
                                      pad_width=(0, 0, 0, 0, mid_padding, mid_padding, mid_padding, mid_padding))
        return mx.nd.concat(upsampling_block, padded_conv_block, dim=1)

    def __init__(self, input_channels, output_channels, **kwargs):
        super(network, self).__init__(**kwargs)
        # convolving
        self.conv_depth0 = self.convoluting_part(input_channels, output_channels=64)
        self.conv_depth1 = self.convoluting_part(64, 128)
        self.conv_depth2 = self.convoluting_part(128, 256)

        # plateau
        self.plateau = self.plateau_block(256, 512)

        # deconvolving
        self.deconv_depth2 = self.deconvoluting_part(512, 256, 128)
        self.deconv_depth1 = self.deconvoluting_part(256, 128, 64)
        self.output_layer = self.output_block(128, 64, output_channels)

    def hybrid_forward(self, F, X):
        conv_block_0 = self.conv_depth0(X)
        conv_block_1 = self.conv_depth1(conv_block_0)
        conv_block_2 = self.conv_depth2(conv_block_1)
        plateau_block_0 = self.plateau(conv_block_2)

        deconv_block_2 = self.concatenate(plateau_block_0, conv_block_2)
        concat_block_2 = self.deconv_depth2(deconv_block_2)

        deconv_block_1 = self.concatenate(concat_block_2, conv_block_1)
        concat_block_1 = self.deconv_depth1(deconv_block_1)

        deconv_block_0 = self.concatenate(concat_block_1, conv_block_0)
        output_layer = self.output_layer(deconv_block_0)
        return output_layer


def show_results(network, features, examples=4):
    figure, axis = plt.subplots(2)

    image_array = network(mx.nd.array(features[0:1], ctx=context).astype('float32')).squeeze(0).asnumpy()

    axis[0].imshow(np.transpose(features[0], (1, 2, 0)))
    print(features[0].shape)
    print(image_array.shape)

    # axis[1].imshow(np.transpose(image_array, (1, 2, 0))[:, :, 0])
    axis[1].imshow(image_array.argmax(0))
    # axis[row][3].imshow(np.transpose(labels[img_idx], (1, 2, 0))[:, :, 0])
    plt.show()




def load_dataset(wi, hi, path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    # print(img.shape)
    transposed_images = np.transpose(img, (0, 1, 2)) / 255.0
    resized_images = resize(transposed_images, (transposed_images.shape[0], wi, hi))
    image = np.zeros((88, resized_images.shape[0], resized_images.shape[1], resized_images.shape[2]))
    for i in range(88):
        image[i] = resized_images
    # print(image.shape)
    return image


# Create a directory in a known location to save files to.
uploads_dir = os.path.join(app.instance_path, 'uploads')
# if os.path.isdir(app.instance_path):
#     os.remove(app.instance_path)
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def home():
   return render_template("index.html")

@app.route('/hello', methods=['GET', 'POST'])
def hello():
   
    # POST request
    if request.method == 'POST':
        # print("HI")
        data = request.files['file']
        data.save(os.path.join(uploads_dir, data.filename))
        # data.save("/")
        path = os.path.join(uploads_dir, data.filename)
        # print(path)
        # img = cv2.imread(path)
        # print(data)
        # path = request.get_json()['file']
        # print(path)
        #
        # path = "NORMAL-1017237-1.jpg"

        (wi, hi), (wo, ho) = (284, 284), (196, 196)
        data = load_dataset(wi, hi, path)

        mynet = network(input_channels=1, output_channels=2)
        mynet.load_parameters('net.params', ctx=context)

        # print(data.shape)
        # show_results(mynet, data)
        image_array = mynet(mx.nd.array(data[0:1], ctx=context).astype('float32')).squeeze(0).asnumpy()
        image_array = image_array.argmax(0)
        main_img = np.transpose(data[0], (1, 2, 0))
        cv2.imwrite("http://127.0.0.3:50000/instance/" + 'main_img.jpeg', main_img)
        cv2.imwrite(uploads_dir + 'image_array.jpeg', image_array)
        retval, buffer = cv2.imencode('.png', main_img[..., 0] * 255)
        pic_str = base64.b64encode(buffer)
        pic_str = pic_str.decode()
        # image = base64.b64encode(np.array(main_img[..., 0])).decode("utf-8")
        retval2, buffer2 = cv2.imencode('.png', image_array * 255)
        # print(main_img[..., 0])
        # print(image_array.shape)
        pic_str2 = base64.b64encode(buffer2)
        pic_str2 = pic_str2.decode()
        img = tf.keras.preprocessing.image.load_img(path , target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        # print(img_batch)

        json_file = open('Dense.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("Dense.h5")
        labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        # labels[np.argmax(loaded_model.predict(img_batch))]
        
        # print(labels[np.argmax(loaded_model.predict(img_batch))])

        return jsonify({'status': True, 'image1': pic_str, 'image2': pic_str2, 'lables': labels[np.argmax(loaded_model.predict(img_batch))]})
        # response = make_response(base64.b64encode(main_img))
        # response.headers.set('Content-Type', 'image/gif')
        # response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
        # return response
        # return send_file(io.BytesIO(main_img),
        #              mimetype='image/jpeg',
        #              as_attachment=True,
        #              attachment_filename='%s.jpg' % "capture")
        #
        # message = {'main': app.instance_path + 'main_img.jpeg',
        #            'mask': app.instance_path + 'image_array.jpeg'}
        # return jsonify(message)  # serialize and use JSON headers

# app.debug = True
# app.run(host="0.0.0.0")
# app.run(host='192.168.43.230', port=50000)
app.run()