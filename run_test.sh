# Test Code (AgeDB-30)
RESOLUTION=0
python test_agedb.py --seed 0 --gpus 0 --data_dir Face/ --down_size $RESOLUTION --mode ir --checkpoint_dir checkpoint/teacher/resol$RESOLUTION-IR

# Test Code (TinyFace)
python test_tinyface.py --seed 0 --gpus 0 --tinyface_dir tinyface/ --mode ir --checkpoint_path checkpoint/student/F_SKD//resol1-IR/last_net.ckpt