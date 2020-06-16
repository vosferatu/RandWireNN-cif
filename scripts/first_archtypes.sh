#!/usr/bin/env bash

for i in {1..10}; do \
   python main.py --name first_gen --seed ${i}; \
done