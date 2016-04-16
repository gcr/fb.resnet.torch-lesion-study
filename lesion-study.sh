#!/bin/bash

for i in {1..60}; do
    OUT=lesion-study-outs/remove-${i}.msgpack
    if [ '!' -f ${OUT} ]; then
        if mkdir ${OUT}-lock; then
            th \
                lesion-study.lua \
                -batchSize 12 \
                -data /mnt/net/data/ \
                -modelPath pretrained/resnet-152.t7 \
                -deleteBlock $i \
                -saveClassAccuracy $OUT
            # rmdir ${OUT}-lock
        fi
    fi
done
