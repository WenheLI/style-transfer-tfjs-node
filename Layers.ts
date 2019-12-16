import * as tf from '@tensorflow/tfjs-node';
import { Tensor4D, model, input, SymbolicTensor } from '@tensorflow/tfjs-node';
import { ConvLayerArgs } from '@tensorflow/tfjs-layers/dist/layers/convolutional';

const {conv2d, batchNormalization, prelu, add, reLU, conv2dTranspose} = tf.layers;
const init = {
    kernelInitializer: tf.initializers.truncatedNormal({stddev: .1, seed: 1})
};
class InstanceNorm extends tf.layers.Layer {
    constructor() {
        super({});
        this.supportsMasking = true;
    }

    call(inputs: Tensor4D, kwargs) {
        const [batch, rows, cols, channels] = inputs.shape;
        const varShape = [channels];
        const {mean, variance} = tf.moments(inputs, [1, 2], true)
        const shift = tf.zeros(varShape);
        const scale = tf.ones(varShape);
        const epsilon = 1e-3;
        const normalized = inputs.sub(mean).div(variance.add(epsilon).pow(.5))
        return normalized.mul(scale).add(shift);
    }

    static get className() {
        return 'InstanceNorm';
    }
}

tf.serialization.registerClass(InstanceNorm);

const instanceNorm = () => {
    return new InstanceNorm();
}

const Conv2dInstanced = (input: tf.SymbolicTensor, args: ConvLayerArgs, relu=true): tf.SymbolicTensor => {

    const tempConv2d = conv2d(Object.assign(init, args)).apply(input);
    const instedTempConv2d = instanceNorm().apply(tempConv2d) as tf.SymbolicTensor;
    if (relu) {
        return reLU().apply(instedTempConv2d) as tf.SymbolicTensor;
    }
    return instedTempConv2d;
}

const ResidualBlock = (input: tf.SymbolicTensor): tf.SymbolicTensor => {
    const tempConv1 = Conv2dInstanced(input, {
        filters: 128,
        kernelSize: 3,
        strides: 1,
        padding: 'same'
    });
    const tempConv2 = Conv2dInstanced(tempConv1, {
        filters: 128,
        kernelSize: 3,
        strides: 1,
        padding: 'same'
    });
    const inputs: Array<tf.SymbolicTensor> = [tempConv1, tempConv2];
    return tf.layers.add().apply(inputs) as tf.SymbolicTensor;
}


const TransConv2dInstanced = (input: tf.SymbolicTensor, args: ConvLayerArgs): tf.SymbolicTensor => {
    const tempTransConv2d = conv2dTranspose(Object.assign(init, args)).apply(input);
    const instedTempTransConv2d = instanceNorm().apply(tempTransConv2d);
    return reLU().apply(instedTempTransConv2d) as tf.SymbolicTensor;
}

export {
    ResidualBlock,
    Conv2dInstanced,
    TransConv2dInstanced
}
