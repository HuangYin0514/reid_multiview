
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


# # v4_IP_mv_2, format
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_12/results
cd /home/hy/project/reid_multiview/v4_IP_mv_12
sh run.sh


# v4_IP_mv_2, format, bn_features = F.normalize(features, p=2, dim=1)
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_13/results
cd /home/hy/project/reid_multiview/v4_IP_mv_13
sh run.sh

# v4_IP_mv_2, format, 共享-指定损失sum方式 
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_14/results
cd /home/hy/project/reid_multiview/v4_IP_mv_14
sh run.sh

# v4_IP_mv_12, 共享/指定特征id约束
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_15/results
cd /home/hy/project/reid_multiview/v4_IP_mv_15
sh run.sh

# v4_IP_mv_15, 共享/指定特征id约束，  共享-指定损失/共享损失
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_16/results
cd /home/hy/project/reid_multiview/v4_IP_mv_16
sh run.sh

# # v4_IP_mv_16, 共享/指定特征id约束，  共享-指定损失/共享损失/指定损失
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_17/results
cd /home/hy/project/reid_multiview/v4_IP_mv_17
sh run.sh