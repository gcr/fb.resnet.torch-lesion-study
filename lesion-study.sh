#!/bin/bash

for i in {1..100}; do
    OUT=lesion-study-outs-200/remove-${i}.msgpack
    if [ '!' -f ${OUT} ]; then
        if mkdir ${OUT}-lock; then
            th \
                lesion-study.lua \
                -batchSize 8 \
                -data /mnt/net/data/ \
                -modelPath pretrained/resnet-200.t7 \
                -deleteBlock $i \
                -saveClassAccuracy $OUT
            # rmdir ${OUT}-lock
        fi
    fi
done
