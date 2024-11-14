
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

# cd /home/hy/project/reid_multiview/v4_IP_mv
# sh vis.sh

#################################################################################
# progress
#################################################################################

# # v4_mv, 修改已有程序的bug, 增加解耦模块（空）
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_2/results
cd /home/hy/project/reid_multiview/v4_IP_mv_2
sh run.sh

# v4_IP_mv_2, 修改整合方式为mean
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_3/results
cd /home/hy/project/reid_multiview/v4_IP_mv_3
sh run.sh

# # v4_IP_mv_2, 神经网络融合
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_4/results
cd /home/hy/project/reid_multiview/v4_IP_mv_4
sh run.sh

