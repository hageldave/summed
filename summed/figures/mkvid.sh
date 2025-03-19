ffmpeg -r 60 -f image2 -s 566x178 -i ./batch/fig_data_%05d.png -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -crf 2  -pix_fmt yuv420p ./data.mp4

