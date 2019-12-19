import { imageLoader } from "./utils"

// import * as tf from '@tensorflow/tfjs-node';
// import * as vgg from './vgg';
// import * as fs from 'fs';

const main = async () => {

    const loader = await imageLoader('./images')
    loader.batch(2).forEachAsync(e => console.log(e.shape))
    // const vgg = await tf.loadLayersModel('file://./models/vgg19/model.json');
    // const contentLayers = ['block5_conv2']
    // const style_layers = ['block1_conv1',
    //             'block2_conv1',
    //             'block3_conv1', 
    //             'block4_conv1', 
    //             'block5_conv1'];
    // const v = await vgg.vggLayers(style_layers);
    // const imgFile = fs.readFileSync('./test.jpg');
    
    // let img = await tf.node.decodeImage(imgFile);
    // img = tf.image.resizeBilinear(img, [224, 224]);
    // img = img.mul(255)
    // // const res = v.predict(img.reshape([1,224,224,3]), {verbose: true});
    // v.layers.forEach((layer) => {
    //     console.log(layer.name)
    //     console.log(layer.apply(img))

    // })

    // const t = vgg.getLayer(style_layers[0])
    // console.log(t)
    // vgg.summary()
}

main()