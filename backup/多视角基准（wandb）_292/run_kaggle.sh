wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

python main.py --mode train --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky --task_name v292 --notes 基准版本 --tags dev baseline

python main.py --mode visualization --output_path results/occluded_duke --occluded_duke_path /kaggle/input/occluded-duke/Occluded_Duke --pid_num 702 --module Lucky
tar -czf occluded_duke_vis.tar.gz results/occluded_duke 
rm -rf results/occluded_duke/actmap results/occluded_duke/ranked_results