/*
Copyright (c) 2014-2015, ArrayFire
Copyright (c) 2015 Gábor Mező aka unbornchikken (gabor.mezo@outlook.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

 * Neither the name of the ArrayFire nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

import Promises from 'bluebird';
import af from '../../..';
import now from 'performance-now';
import _ from 'lodash';

const async = Promises.coroutine;

let runOnDevices = async(function*(platformID, f, onID) {
    let afOfPlatform = af(platformID);
    for (let deviceInfo of afOfPlatform.getDevices()) {
        if (_.isUndefined(onID) || onID === deviceInfo.id) {
            afOfPlatform.setDevice(deviceInfo.id);
            const start = now();
            yield f(afOfPlatform, deviceInfo);
            const end = now();
            console.log(`\n-- took ${((end - start) / 1000).toFixed(10)} seconds\n`);
        }
    }
});

let runOnAllPlatforms = async(function*(f, name) {
    let platfroms = af.supportedPlatforms();
    console.log(`Running ${name} on all supported platfroms: ${platfroms.join(', ')}\n`);
    try {
        for (let id of platfroms) {
            yield runOnDevices(id, f);
        }
    }
    catch (e) {
        console.error(e.stack);
    }
});

let runOnBestDevice = async(function*(f, name) {
    let platfroms = af.supportedPlatforms();
    let order = ['CUDA', 'OpenCL', 'CPU'];
    console.log(`Running ${name} on best available device.\n`);
    try {
        if (_(platfroms).contains(order[0])) {
            yield runOnDevices(order[0], f, 0);
        }
        else if (_(platfroms).contains(order[1])) {
            yield runOnDevices(order[1], f, 0);
        }
        else {
            yield runOnDevices(order[2], f, 0);
        }
    }
    catch (e) {
        console.error(e.stack);
    }

    try {

    } catch (e) {

    }
});

let printDeviceInfo = function (deviceInfo) {
    console.log(`ID: ${deviceInfo.id}\nName: ${deviceInfo.name}\nPlatform: ${deviceInfo.platform}\nToolkit: ${deviceInfo.toolkit}\nCompute: ${deviceInfo.compute}`);
};

module.exports = {
    runOnAllPlatforms: runOnAllPlatforms,
    runOnBestDevice: runOnBestDevice,
    printDeviceInfo: printDeviceInfo
};
