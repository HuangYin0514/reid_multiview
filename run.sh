
#################################################################################
# Common
#################################################################################
# ssh hy@10.244.37.93
# screen -D -r training
# conda activate py396
# cd /home/hy/project/reid_multiview/
# sh run.sh

#################################################################################
# copy
#################################################################################
# cp -rf /home/hy/project/reid_multiview/v3_backbone_IP /home/hy/project/reid_multiview/v3_backbone_IP_tsne

#################################################################################
# Vis
#################################################################################
# cd /home/hy/project/reid_multiview/v3_backbone_IP
# sh vis.sh

# cd /home/hy/project/reid_multiview/v3_backbone_IP_tsne
# sh vis.sh

#################################################################################
# progress
#################################################################################
# backbone
# rm -rf /home/hy/project/reid_multiview/v3_backbone_IP/results
# cd /home/hy/project/reid_multiview/v3_backbone_IP
# sh run.sh

# seam
rm -rf /home/hy/project/reid_multiview/v4_seam/results
cd /home/hy/project/reid_multiview/v4_seam
sh run.sh
