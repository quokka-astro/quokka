#!/bin/sh

ffmpeg -framerate 30 -pattern_type glob -i "seq1/*.png" -framerate 30 -pattern_type glob -i "seq2/*.png" \
        -filter_complex "\
        [0:v]setpts=PTS-STARTPTS,scale=-1:ih[left]; \
        [1:v]setpts=PTS-STARTPTS,scale=-1:ih[right]; \
        [left][right]hstack=inputs=2[out]" \
        -map "[out]" -r 30 -vcodec libx264 -pix_fmt yuv420p -preset slow -tune animation -crf 18 disk_movie_combined.mp4
