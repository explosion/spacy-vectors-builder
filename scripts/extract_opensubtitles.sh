#!/bin/sh

zcat $1 | egrep -v '^[(].*[)]$' | sed "s/´/'/g" > $2
