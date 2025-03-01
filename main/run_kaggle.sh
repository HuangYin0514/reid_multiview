###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

###########################################################
# Runs
###########################################################
python main.py --task_name v430 --notes R-IR-APDFI --tags dev resnet50 occluded_duke --mode train --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky 


# python main.py --task_name test --notes mac实验新方法 --tags mac dev resnet50 occluded_duke --mode train --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky 

###########################################################
# Visualize
###########################################################
# python main.py --mode visualization --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky
# tar -czf occluded_duke_vis.tar.gz results/occluded_duke 
# rm -rf results/occluded_duke/actmap results/occluded_duke/ranked_results