import debug from 'debug';
import now from 'performance-now';

debug('af:ann');

export default class ANN {
  constructor(af, layers, range) {
    range = range || 0.05;
    this.af = af;
    this.numLayers = layers.length;
    this.signal = [];
    this.weights = [];
    for (let i = 0; i < this.numLayers; i++) {
        this.signal.push(new af.AFArray());
        if (i < this.numLayers - 1) {
            let w = af
              .randu(layers[i] + 1, layers[i + 1], af.dType.f32)
              .mul(range)
              .sub(range / 2);
            this.weights.push(w);
        }
    }
  }

  deriv(out) {
    return out
      .rhsSub(1)
      .mul(out);
  }

  addBias(input) {
    let { af } = this;
    return af.join(1, af.constant(1, input.dims(0), af.dType.f32), input);
  }

  _calculateError(out, pred) {
    let dif = out.sub(pred);
    let sq = dif.mul(dif);
    let { af } = this;
    return Math.sqrt(af.sum(sq)) / sq.elements();
  }

  forwardPropagate(input) {
    let { af } = this;
    this.signal[0].set(input);

    for (let i = 0; i < this.numLayers - 1; i++) {
      af.scope(() => {
        let inVec = this.addBias(this.signal[i]);
        let outVec = af.matMul(inVec, this.weights[i]);
        this.signal[i + 1].set(af.sigmoid(outVec));
      });
    }
  }

  backPropagate(target, alpha) {
    let { af } = this;
    let { Seq } = af;

    // Get error for output layer
    af.scope(() => {
      let outVec = this.signal[this.numLayers - 1];
      let err = outVec.sub(target);
      let m = target.dims(0);

      for (let i = this.numLayers - 2; i >= 0; i--) {
        af.scope(() => {
          let inVec = this.addBias(this.signal[i]);
          let delta = af.transpose(
            this
              .deriv(outVec)
              .mul(err)
          );

          // Adjust weights
          let grad = af.matMul(delta, inVec)
            .mul(alpha)
            .neg()
            .div(m);

          this.weights[i].addAssign(af.transpose(grad));

          // Input to current layer is output of previous
          outVec = this.signal[i];
          err.set(af.matMulTT(delta, this.weights[i]));

          // Remove the error of bias and propagate backward
          err.set(err.at(af.span, new Seq(1, outVec.dims(1))));
        });
      }
    });
  }

  predict(input) {
    this.forwardPropagate(input);
    return this.signal[this.numLayers - 1].copy();
  }

  train(input, target, options) {
    let { af } = this;
    let { Seq } = af;

    let numSamples = input.dims(0);
    let numBatches = numSamples / options.batchSize;

    let err = 0;
    let allTime = 0;

    for (let i = 0; i < options.maxEpochs; i++) {
      const start = now();
      for (let j = 0; j < numBatches - 1; j++) {
        af.scope(() => {
          let startPos = j * options.batchSize;
          let endPos = startPos + options.batchSize - 1;

          let x = input.at(new Seq(startPos, endPos), af.span);
          let y = target.at(new Seq(startPos, endPos), af.span);

          this.forwardPropagate(x);
          this.backPropagate(y, options.alpha);
        });
      }

      af.scope(() => {
        // Validate with last batch
        let startPos = (numBatches - 1) * options.batchSize;
        let endPos = numSamples - 1;
        let outVec = this.predict(input.at(new Seq(startPos, endPos), af.span));
        err = this._calculateError(outVec, target.at(new Seq(startPos, endPos), af.span));
      });

      const end = now();
      allTime += (end - start) / 1000;

      if ((i + 1) % 10 === 0) {
        console.log(`Epoch: ${i + 1}, Error: ${err.toFixed(6)}, Duration: ${(allTime / 10).toFixed(6)} seconds`);
        allTime = 0;
      }

      // Check if convergence criteria has been met
      if (err < options.maxError) {
        console.log(`Converged on Epoch: ${i + 1}`);
        break;
      }
    }

    return err;
  }
}
