
# make frames
shellutils.py convert_image -i frame_???0.pdf -ot jpg

# make cropped video
shellutils.py make_movie -i *.jpg -o tg.mp4 -c mpeg4 -b 5000k -ffmpeg_args -vf '"[in] crop=860:560:190:90 [out]"'

# reverse video
shellutils.py make_movie -i tg.mp4 -o tgr.mp4 -c mpeg4 -b 5000k -ffmpeg_args -vf 'reverse'

# hold last frame for 10s
shellutils.py make_movie -i tgr.mp4 -o tgrh.mp4 -c mpeg4 -b 5000k -ffmpeg_args -vf 'tpad=stop_mode=clone:stop_duration=10'

# concatenate
ffmpeg -i tgrh.mp4 -i tg.mp4 -filter_complex "concat=n=2:v=1:a=0" TurbGen.mp4
