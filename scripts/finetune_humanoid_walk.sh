task_name='humanoid_walk'
n_traj=$1

cmd="python train_bc.py task_name=${task_name} seed=0 exp_name=${task_name}_${n_traj}_NEWDATA2 n_traj=${n_traj}"
cmd="$cmd encoder_dir=/data2/seongwoongjo/premier-taco/pretrained_ckpt/encoder.pt"
cmd="$cmd offline_data_dir=/data2/bestgenius10/premier-taco/PTACO_DATASET/dmc_eval_data/${task_name}"

echo $cmd
eval $cmd
