
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


# # # v4_IP_mv_3, seam注意力
# rm -rf /home/hy/project/reid_multiview/v4_IP_mv_4/results
# cd /home/hy/project/reid_multiview/v4_IP_mv_4
# sh run.sh

# # #  # 修改了对比损失函数
# rm -rf /home/hy/project/reid_multiview/v4_IP_mv_5/results
# cd /home/hy/project/reid_multiview/v4_IP_mv_5
# sh run.sh

# #  # v4_IP_mv_5, special view fusion
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_6/results
cd /home/hy/project/reid_multiview/v4_IP_mv_6
sh run.sh

# #  # v4_IP_mv_6, 积分方式为mean
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_6/results
cd /home/hy/project/reid_multiview/v4_IP_mv_6
sh run.sh