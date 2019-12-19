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

export function net(data, input) {
    const layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 
        'pool1', 'conv2_1', 'relu2_1', 
        'conv2_2', 'relu2_2', 'pool2', 
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    ];


}

export function preprocess(x: tf.Tensor) {
    return x.sub(MEAN_PIXEL);
}

export function unprocess(x: tf.Tensor) {
    return x.add(MEAN_PIXEL);
}