# Test Code (AgeDB-30)
RESOLUTION=0
python test_agedb.py --seed 0 --gpus 0 --data_dir Face/ --down_size $RESOLUTION --mode ir --checkpoint_dir checkpoint/teacher/resol$RESOLUTION-IR

# Test Code (TinyFace)
python test_tinyface.py --gpus 0 --tinyface_dir aligned_pad_0.1_pad_high/ --mode ir --save_dir result/ --checkpoint_path checkpoint/student/F_SKD/resol1-IR/last_net.ckpt