# Baseline Network for {multi, 2x, 4x, 8x}
for RESOLUTION in 1 56 28 14
do
    # IR - Down
    python train_teacher.py --seed 0 --gpus 1 --data_dir Face/ --down_size $RESOLUTION --mode ir --margin_type CosFace --save_dir checkpoint/student/baseline/resol$RESOLUTION-IR

    # CBAM - Down
    python train_teacher.py --seed 0 --gpus 1 --data_dir Face/ --down_size $RESOLUTION --mode cbam --margin_type CosFace --save_dir checkpoint/student/baseline/resol$RESOLUTION-CBAM
done


# F-SKD Network for {multi, 2x, 4x, 8x}
for RESOLUTION in 1 56 28 14
do
    D_PARAM=20.0

    # IR - Down
    teacher=checkpoint/teacher/resol0-IR/last_net.ckpt
    python train_student.py --seed 0 --gpus 0 --data_dir Face/ --down_size $RESOLUTION --mode ir --margin_type CosFace --distill_type F_SKD --distill_param $D_PARAM --save_dir checkpoint/student/F_SKD/resol$RESOLUTION-IR --teacher_path $teacher

    # CBAM - Down
    teacher=checkpoint/teacher/resol0-CBAM/last_net.ckpt
    python train_student.py --seed 0 --gpus 0 --data_dir Face/ --down_size $RESOLUTION --mode cbam --margin_type CosFace --distill_type F_SKD --distill_param $D_PARAM --save_dir checkpoint/student/F_SKD/resol$RESOLUTION-CBAM --teacher_path $teacher
done