import * as tf from '@tensorflow/tfjs';

const MEAN_PIXEL = tf.tensor([123.68, 116.779, 103.939]);

export function preprocess(x: tf.Tensor) {
    return x.sub(MEAN_PIXEL);
}