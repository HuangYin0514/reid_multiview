###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

###########################################################
# Runs
###########################################################
python main.py --config_file "config/method.yml" 

# python main.py --config_file "config/occluded_reid/0.yml" 
# python main.py --config_file "config/occluded_reid/1.yml" 
# python main.py --config_file "config/occluded_reid/2.yml" 
# python main.py --config_file "config/occluded_reid/3.yml" 
# python main.py --config_file "config/occluded_reid/4.yml"
# python main.py --config_file "config/occluded_reid/5.yml"
# python main.py --config_file "config/occluded_reid/6.yml"
# python main.py --config_file "config/occluded_reid/7.yml"
# python main.py --config_file "config/occluded_reid/8.yml"
# python main.py --config_file "config/occluded_reid/9.yml"

###########################################################
# Visualize
###########################################################
# python main.py --mode visualization --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky
# tar -czf occluded_duke_vis.tar.gz results/occluded_duke 
# rm -rf results/occluded_duke/actmap results/occluded_duke/ranked_results