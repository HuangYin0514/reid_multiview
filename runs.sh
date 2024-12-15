
#!/bin/bash

#################################################################################
# Common
#################################################################################
# ssh hy@10.244.37.93
# screen -D -r training
# screen -S training
# conda activate py396
# cd /home/hy/project/reid_multiview/
# sh runs.sh

#################################################################################
# copy
#################################################################################
# cp -rf /home/hy/project/reid_multiview/v3_backbone_IP /home/hy/project/reid_multiview/v3_backbone_IP_tsne

#################################################################################
# Vis
#################################################################################
# cd /home/hy/project/reid_multiview/v3_backbone_IP
# sh vis.sh

# cd /home/hy/project/reid_multiview/v4_IP_mv
# sh vis.sh

#################################################################################
# progress (Linux)
#################################################################################

# #注意空格
root_folder="/home/hy/project/reid_multiview/$folder"
folder="v4_IP_mv_1"
# folder="v4_IP_mv_2"
# folder="v4_IP_mv_3"
echo $root_folder/$folder

rm -rf $root_folder/$folder/results
cd $root_folder/$folder
sh run.sh