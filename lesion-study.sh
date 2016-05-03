#!/bin/bash

TAG=$1
if [ -z "$TAG" ]; then
    echo "Usage: $0 tag"
    exit 1
fi

for i in {1..100}; do
    DEST_PATH=lesion-study-outs-cifar10/$TAG
    # For stochastic depth models trained using Sam's preresnet code:
    # MODEL=trained_models/model_${TAG}.t7
    # For stochastic depth models trained using the Stochastic_Depth code:
    MODEL=Stochastic_Depth/results/model_${TAG}.t7
    mkdir -p $DEST_PATH
    OUT=${DEST_PATH}/remove-${i}.msgpack
    if [ '!' -f ${OUT} ]; then
        if mkdir ${OUT}-lock; then
            # For CIFAR-10:
            th \
                lesion-study.lua \
                -batchSize 128 \
                -dataset cifar10 \
                -modelPath ${MODEL} \
                -deleteBlock $i \
                -saveClassAccuracy ${OUT}
            # For imagenet:
            # th \
            #     lesion-study.lua \
            #     -batchSize 8 \
            #     -data /mnt/net/data/ \
            #     -modelPath pretrained/resnet-200.t7 \
            #     -deleteBlock $i \
            #     -saveClassAccuracy $OUT
            # rmdir ${OUT}-lock
        fi
    fi
done
