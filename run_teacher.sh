################################# Teacher ########################################
# teacher network with iResNet50-IR
RESOLUTION=0
python train_teacher.py --seed 0 --gpus 0 --data_dir Face/ --down_size $RESOLUTION --mode ir --margin_type CosFace --save_dir checkpoint/teacher/resol$RESOLUTION-IR

# teacher network with iResNet50-CBAM
RESOLUTION=0
python train_teacher.py --seed 0 --gpus 1 --data_dir Face/ --down_size $RESOLUTION --mode cbam --margin_type CosFace --save_dir checkpoint/teacher/resol$RESOLUTION-CBAM