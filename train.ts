import * as tf from '@tensorflow/tfjs';
import {preprocess} from './vgg';

const gramMatrix = (input: tf.Tensor) => {
    const [batch, width, height, channel] = input.shape;
    const x = input.reshape([batch * channel, -1]);
    return tf.matMul(x, x.transpose());
}

const styleLoss = (input: tf.Tensor) => {
    let loss = 0;

}

const fit = async (model: tf.LayersModel, dataset: tf.data.Dataset<tf.TensorContainer>) => {
    const optimizer = tf.train.rmsprop(.001);
    await dataset.forEachAsync((it) => {
        optimizer.minimize((): tf.Tensor => {
            it.
            preprocess(it)
        })
    });
}

// const optimizer = tf.train.sgd(0.1 /* learningRate */);
// // Train for 5 epochs.
// for (let epoch = 0; epoch < 5; epoch++) {
//   await ds.forEachAsync(({xs, ys}) => {
//     optimizer.minimize(() => {
//       const predYs = model(xs);
//       const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
//       loss.data().then(l => console.log('Loss', l));
//       return loss;
//     });
//   });
//   console.log('Epoch', epoch);
// }