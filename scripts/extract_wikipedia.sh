#!/bin/sh

bunzip2 -c $1 | stack exec wiki2text > $2
