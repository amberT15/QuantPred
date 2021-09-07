cell_line=8
# model_paths='run-20210626_193453-nwu1npo6','run-20210626_193322-o3xjln3i','run-20210626_225543-7sculd4q','run-20210626_224850-uedagrt5'
run_path=/mnt/31dac31c-c4e2-4704-97bd-0788af37c5eb/shush/wandb/wandb_elzar/run-20210828_063923-h3rrpt6i/files
# data_path=/mnt/1a18a49e-9a31-4dbf-accd-3fb8abbfab2d/shush/4grid_atac/lite/random_chop/i_3072_w_1/

CUDA_VISIBLE_DEVICES=2 python ./get_preds.py $cell_line $run_path
./dash_predictions.py $cell_line $run_path
