from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os

import vgg_preprocessing

tf.app.flags.DEFINE_string('output_directory', '/home/brijesh/Documents/TensorFlow', 'Output data directory')

FLAG = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

_DEFAULT_IMAGE_SIZE = 250
_NUM_CHANNELS = 3
_NUM_CLASSES = 2
_NUM_TRAIN_FILES = 640


def parse_record(raw_record, is_training):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]),
        _NUM_CHANNELS)

    # Note that tf.image.convert_image_dtype scales the image data to [0, 1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=is_training)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32)

    return {"image": image}, label


def input_fn(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

    dataset = dataset.map(lambda value: parse_record(value, is_training),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()

    return features, labels


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    f = features["image"]
    input_layer = tf.reshape(f, [100, 299, 299, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [100, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def parse(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.
    return image, label


def get_file_lists(data_dir):
    import glob

    train_list = glob.glob(data_dir + '/' + 'train-*')
    valid_list = glob.glob(data_dir + '/' + 'validation-*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list


def main(argv=None):
    train_list, valid_list = get_file_lists(FLAG.output_directory)

    img_size = _DEFAULT_IMAGE_SIZE
    num_channel = 3
    img_shape = [img_size, img_size, num_channel]

    num_classes = 4
    feature_image = tf.feature_column.numeric_column("image",
                                                     img_shape)

    feature_columns = [feature_image]

    num_hidden_units = [512, 256, 128]

    model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=num_hidden_units,
                                       activation_fn=tf.nn.relu,
                                       n_classes=num_classes,
                                       model_dir=os.path.join(FLAG.output_directory, "tb"))

    model.train(input_fn=lambda: train_input_fn(train_list), steps=200)


def train_input_fn(file_path):
    return input_fn(True, file_path, 50, None, 10)


# is_training, filenames, batch_size, num_epochs=1,num_parallel_calls=1
def test_input_fn(file_path):
    return input_fn(False, file_path, 50, 1, 1)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()