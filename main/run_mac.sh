###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

###########################################################
# Runs: sh /Users/drhy/Documents/project/reid_multiview/main/run_mac.sh
###########################################################
cd /Users/drhy/Documents/project/reid_multiview/main
rm -rf results/occluded_duke

python main.py --task_name test --notes mac实验新方法 --tags mac dev resnet50 occluded_duke --module Lucky --mode train --output_path results/occluded_duke --occluded_duke_path /Users/drhy/Documents/project/reid_multiview/v1_dataset_processing/Occluded_Duke/Occluded_Duke_mac --pid_num 702 --cuda cpu --batchsize 12 --eval_epoch 1 --total_train_epoch 1


###########################################################
# Visualize
###########################################################
# python main.py --mode visualization --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky
# tar -czf occluded_duke_vis.tar.gz results/occluded_duke 
# rm -rf results/occluded_duke/actmap results/occluded_duke/ranked_results