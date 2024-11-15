
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

# # v4_mv,  解耦共享信息和指定信息
# rm -rf /home/hy/project/reid_multiview/v4_IP_mv_2/results
# cd /home/hy/project/reid_multiview/v4_IP_mv_2
# sh run.sh

v4_IP_mv_2, 共享特征损失
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_3/results
cd /home/hy/project/reid_multiview/v4_IP_mv_3
sh run.sh

# v4_IP_mv_2,  只用共享特征进行识别
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_4/results
cd /home/hy/project/reid_multiview/v4_IP_mv_4
sh run.sh

# v4_IP_mv_2, sharedSpecialLoss + sharedLoss
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_5/results
cd /home/hy/project/reid_multiview/v4_IP_mv_5
sh run.sh

# # v4_IP_mv_2,  共享、特殊、共享特殊
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_6/results
cd /home/hy/project/reid_multiview/v4_IP_mv_6
sh run.sh
