import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';

function* imageLoaderGenerator(files: Array<string>, path: string) {
    const totalElements = files.length;
    console.log(`Toal Elements are ${totalElements}`);
    let cursor = 0;
    while (cursor < totalElements) {
        const f = fs.readFileSync(`${path}/${files[cursor]}`);
        cursor += 1;
        const temp = tf.node.decodeImage(f)
        yield temp;
    }
}

export function imageLoader(path: string): tf.data.Dataset<tf.Tensor4D> {
    const files = fs.readdirSync(path);
    return tf.data.generator(imageLoaderGenerator.bind(this, files, path))
}