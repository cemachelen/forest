#!/bin/bash
set -x
SCRIPT_DIR=$(dirname $0)

FOREST_CONFIG=${SCRIPT_DIR}/highway.yaml \
FOREST_DIR=~/buckets/stephen-sea-public-london/ \
    bokeh serve ${SCRIPT_DIR}/ --dev
