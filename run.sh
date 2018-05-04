#! /bin/bash
mkdir -p /s3/stephen-sea-public-london
goofys stephen-sea-public-london /s3/stephen-sea-public-london

if [ -z "$PUBLIC_IP" ]
then
    bokeh serve --port 8888 /opt/model_evaluation_tool/documents/*
else
    bokeh serve  --host $PUBLIC_IP:8888 --allow-websocket-origin $PUBLIC_IP:8888 --port 8888  /opt/model_evaluation_tool/documents/*
fi