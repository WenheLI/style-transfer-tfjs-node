import * as tf from '@tensorflow/tfjs-node';
import * as vgg from './vgg';
const main = async () => {
    // const vgg = await tf.loadLayersModel('file://./models/vgg19/model.json');
    const contentLayers = ['block5_conv2']
    const style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'];
    const v = await vgg.vggLayers(style_layers);
    v.predict(tf.ones([1, 224, 224, 3])).print()
    // const t = vgg.getLayer(style_layers[0])
    // console.log(t)
    // vgg.summary()
}

main()