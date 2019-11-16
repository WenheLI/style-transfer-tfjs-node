import * as tf from '@tensorflow/tfjs-node';
import { ResidualBlock, Conv2dInstanced, TransConv2dInstanced } from "./Layers";

const {activation} = tf.layers;

const input = tf.input({shape: [640, 480, 3]});

const conv2d_1 = Conv2dInstanced(input, {
    inputShape: [640, 480, 3],
    filters: 32,
    kernelSize: 9,
    strides: 1,
    padding: "same"
});

console.log(conv2d_1)

const conv2d_2 = Conv2dInstanced(conv2d_1, {
    filters: 64,
    kernelSize: 3,
    strides: 2,
    padding: "same"
});

const conv2d_3 = Conv2dInstanced(conv2d_2, {
    filters: 128,
    kernelSize: 3,
    strides: 2,
    padding: "same"
});
const resi_1 = ResidualBlock(conv2d_3);
const resi_2 = ResidualBlock(resi_1);
const resi_3 = ResidualBlock(resi_2);
const resi_4 = ResidualBlock(resi_3);
const resi_5 = ResidualBlock(resi_4);
const trans_1 = TransConv2dInstanced(resi_5, {
    filters: 64,
    kernelSize: 3,
    strides: 2,
    padding: 'same'
});

const trans_2 = TransConv2dInstanced(trans_1, {
    filters: 32,
    kernelSize: 3,
    strides: 2,
    padding: 'same'
});

const trans_3 = Conv2dInstanced(trans_2, {
    filters: 3,
    kernelSize: 9, 
    strides: 1
}, false);

let output = activation({activation: 'tanh'}).apply(trans_3) as tf.SymbolicTensor;

const model = tf.model({inputs: input, outputs: output});

export default model;