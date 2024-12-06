
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
# progress
#################################################################################

# rm -rf /home/hy/project/reid_multiview/v4_IP_mv_5/results
# cd /home/hy/project/reid_multiview/v4_IP_mv_5
# sh run.sh

#注意空格
folder="v4_IP_mv_1"
folder="v4_IP_mv_2"
folder="v4_IP_mv_3"
# folder="v4_IP_mv_4"
# folder="v4_IP_mv_5"
# folder="v4_IP_mv_6"
# folder="v4_IP_mv_7"
# folder="v4_IP_mv_8"
echo $folder  

rm -rf /home/hy/project/reid_multiview/$folder/results
cd /home/hy/project/reid_multiview/$folder
sh run.sh





