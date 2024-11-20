
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


# # IP结合注意力
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_1/results
cd /home/hy/project/reid_multiview/v4_IP_mv_1
sh run.sh
