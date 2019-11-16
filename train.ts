import * as tf from '@tensorflow/tfjs';
import {preprocess} from './vgg';
const fit = async (dataset: tf.data.Dataset<tf.TensorContainer>) => {

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