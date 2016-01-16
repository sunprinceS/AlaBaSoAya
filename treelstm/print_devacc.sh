#!/bin/bash

cat $1 | grep 'dev score' | cut -d ' ' -f 4
