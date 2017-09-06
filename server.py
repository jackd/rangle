import tensorflow as tf
from estimator import ModelBuilder, get_model_dir


class TfServer(object):
    def __init__(self, model_name):
        self._model_name = model_name

    def build_graph(self):
        builder = ModelBuilder(self._model_name)
        self._image = tf.placeholder(
            shape=(64, 64, 3), dtype=tf.uint8, name='image')
        float_image = tf.per_image_standardization(self._image)
        float_image = tf.expand_dims(float_image, axis=0)
        self._inference = tf.squeeze(builder.get_inference(
            float_image, training=False), axis=0)
        print('Graph built')

    def init_sess(self):
        self._sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self._sess, tf.train.latest_checkpoint(
            get_model_dir(self._model_name)))
        print('Session restored')

    def serve(self, image):
        if image.shape != (64, 64, 3):
            raise Exception(
                'Shape must be 64 * 64 * 3, got %s' % str(image.shape))
        inf = self._sess.run(self._inference, feed_dict={self._image: image})
        return inf

    def close(self):
        self._sess.close()


if __name__ == '__main__':
    def load_inference_image():
        raise NotImplementedError()

    model_name = 'base'
    server = TfServer(model_name)
    server.build_graph()
    server.init_sess()
    my_image = load_inference_image()
    assert(my_image == (64, 64, 3))
    inference = server.serve(my_image)
    print('Model infers: %s' % str(inference))

    server.close()
