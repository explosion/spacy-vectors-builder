#!/bin/sh

if [ ! -d $1 ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone $2 $1
fi 

cd $1

git checkout $3

git lfs pull --include "packaged/$4/*"
