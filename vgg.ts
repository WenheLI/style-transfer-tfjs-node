import * as tf from '@tensorflow/tfjs-node';

const MEAN_PIXEL = tf.tensor([123.68, 116.779, 103.939]);

export const vggLayers = async (layerNames: Array<string>) => {
    const vgg = await tf.loadLayersModel('file://./models/vgg19/model.json');
    vgg.trainable = false;
    const inputs = vgg.input;
    let outputs = inputs;
    layerNames.forEach((it) => {
        outputs = vgg.getLayer(it).apply(outputs) as tf.SymbolicTensor;
    })
    const model = tf.model({inputs , outputs});
    model.summary();
    return model;
}

export function preprocess(x: tf.Tensor) {
    return x.sub(MEAN_PIXEL);
}