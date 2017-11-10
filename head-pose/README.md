# Wrie all file names in a .txt file

```sh
python3 write_frame_jpg_file_names_in_txt_file.py -sW ABSOLUTELY -eW ABUSE
```

# Run deepgaze to estimate head_pose

Deep gaze uses CNN to estimate roll, pitch, yaw.

[deepgaze GitHub](https://github.com/mpatacchiola/deepgaze)

[pdf](https://mpatacchiola.github.io/doc/Head%20Pose%20Estimation%20in%20the%20Wild%20using%20Convolutional%20Neural%20Networks%20and%20Adaptive%20Gradient%20Methods%20-%20Patacchiola%20and%20Cangelosi%20-%202017.pdf)

[My fork of deepgaze](https://github.com/voletiv/deepgaze)

```sh
python3 estimate_head_pose_for_images_list.py -sW ABSOL -sE YEARS
```

## - OR -

# Run gazr_benchmark_head_pose_multiple_frames

```python
from head_pose_functions import *
run_dlib_head_pose_estimator(LRW_DATA_DIR)
```
