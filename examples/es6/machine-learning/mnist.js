import Promises from 'bluebird';
import ref from 'ref';
import fsExtra from 'fs-extra';
import debug from 'debug';
import _ from 'lodash';
import assert from 'better-assert';
import path from 'path';

debug('af:mnist');

const async = Promises.coroutine;
const fs = Promises.promisifyAll(fsExtra);
const float = ref.types.float;
const uint = ref.types.uint;

const readData = async(function*(f, data) {
    let bytesRead = yield fs.readAsync(f, data, 0, data.length, null);
    if (bytesRead !== data.length) {
        throw new Error('File reading error!');
    }
});

const readIdx = async(function*(path, type) {
    let file = yield fs.openAsync(path, 'r');
    try {
        let d = new Buffer(4);

        yield readData(file, d);

        if (d[2] != 8) {
            throw new Error('Unsupported data type');
        }

        const numDims = d[3];

        // Read the dimensions
        let elem = 1;
        let dims = [];
        for (let i = 0; i < numDims; i++) {
            yield readData(file, d);
            let dim = d.readUInt32BE();
            elem *= dim;
            dims.push(dim);
        }

        // Read the data
        let bdata = new Buffer(elem);
        yield readData(file, bdata);
        let data = new Buffer(bdata.length * type.size);
        for (let i = 0; i < bdata.length; i++) {
            let byte = bdata[i];
            type.set(data, i * type.size, byte);
        }

        return {
            numDims: numDims,
            dims: dims,
            data: data
        };
    } finally {
        yield fs.closeAsync(file);
    }
});

export default const mnist = {
    setup: async(function*(af, expandLabels, frac) {
        frac = Math.min(frac || 1.0, 0.8);
        let dataRoot =  path.resolve(path.join(__dirname, '../../ml_lab/data/mnist'));
        let AFArray = af.AFArray;
        let Dim4 = af.Dim4;

        let imageData = yield readIdx(path.join(dataRoot, 'images-subset'), float);
        let labelData = yield readIdx(path.join(dataRoot, 'labels-subset'), uint);

        let rIDims = new Dim4(_(imageData.dims).reverse().value());
        let images = yield AFArray.createAsync(rIDims, af.dType.f32, imageData.data);

        let r = af.randu(10000, af.dType.f32);
        let cond = r.lt(frac);
        let trainIndices = af.where(cond);
        let testIndices = af.where(cond.not());

        let trainImages = af.lookup(images, trainIndices, 2).div(255);
        let testImages = af.lookup(images, testIndices, 2).div(255);

        let numClasses = 10;
        let numTrain = trainImages.dims(2);
        let numTest = testImages.dims(2);

        debug(`Training sample count: ${numTrain}`);
        debug(`Test sample count: ${numTest}`);

        let trainLabels;
        let testLabels;

        if (expandLabels) {
            trainLabels = af.constant(0, numClasses, numTrain, af.dType.f32);
            testLabels = af.constant(0, numClasses, numTest, af.dType.f32);

            assert(trainIndices.type() === af.dType.u32);
            assert(testIndices.type() === af.dType.u32);

            let hTrainIdx = yield trainIndices.hostAsync();
            let hTestIdx = yield testIndices.hostAsync();

            for (let i = 0; i < numTrain; i++) {
                let idx = uint.get(hTrainIdx, i * uint.size);
                let label = uint.get(labelData.data, idx * uint.size);
                assert(label >= 0 && label <= 9);
                trainLabels.set(label, i, 1);
            }

            for (let i = 0; i < numTest; i++) {
                let idx = uint.get(hTestIdx, i * uint.size);
                let label = uint.get(labelData.data, idx * uint.size);
                assert(label >= 0 && label <= 9);
                testLabels.set(label, i, 1);
            }
        } else {
            let labels = yield AFArray.createAsync(labelData.dims[0], af.dType.u32, labelData.data);
            trainLabels = labels.at(trainIndices);
            testLabels = labels.at(testIndices);
        }

        return {
            numClasses: numClasses,
            numTrain: numTrain,
            numTest: numTest,
            trainImages: trainImages,
            testImages: testImages,
            trainLabels: trainLabels,
            testLabels: testLabels
        };
    })
};
