#!/bin/sh

CONVERT_EXE=/sw/andes/spack-envs/base/opt/linux-rhel8-x86_64/gcc-8.3.1/imagemagick-7.0.8-7-srnwz5bn26ielesvto2lrctsw4qby573/bin/convert
RENAME_EXE=/gpfs/alpine2/ast196/proj-shared/wibking/spack/opt/spack/linux-rhel8-zen2/gcc-9.3.0/rename-1.600-jzccmpqqd2tu2muki2fd3o2aehxkdvr4/bin/rename
FFMPEG_EXE=/gpfs/alpine2/ast196/proj-shared/wibking/spack/opt/spack/linux-rhel8-zen2/gcc-9.3.0/ffmpeg-7.0.2-ywl5f5ldeogjs3jtgitdufzpksnswgit/bin/ffmpeg

set -x

## rename *.png files

$RENAME_EXE 's/\d+/sprintf("%07d",$&)/e' -- *.png

## resize frames

for file in *.png;
do
    $CONVERT_EXE $file -resize 2048x2048 $(basename -s .png $file).resize_large.png
done

## make movie
FPS=10
FILTERS="pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white, drawtext=fontfile=/usr/share/fonts/google-droid/DroidSans.ttf:text='QED-QUOKKA Simulation':fontcolor=white:fontsize=48:box=1:boxcolor=black@0.6:boxborderw=5:x=0.9*(w-text_w):y=0.9*(h-text_h)"

$FFMPEG_EXE -framerate $FPS -pattern_type glob -i "slices_*.resize_large.png" -r $FPS -vcodec libx264 -vf "$FILTERS" -pix_fmt yuv420p -preset slow -tune animation -crf 18 slice_movie_large.mov
