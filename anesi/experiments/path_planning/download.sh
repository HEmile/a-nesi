#!/usr/bin/env bash

# From https://github.com/nec-research/tf-imle

mkdir -p data/
cd data/

rm -rf warcraft_shortest_path
wget -c http://data.neuralnoise.com/warcraft_maps.tar.gz
tar xvfz warcraft_maps.tar.gz
mv warcraft_shortest_path_oneskin warcraft_shortest_path
rm -f warcraft_maps.tar.gz