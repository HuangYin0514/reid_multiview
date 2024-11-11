
#################################################################################
# Common
#################################################################################
# ssh hy@10.244.37.93
# screen -D -r training
# conda activate py396
# cd /home/hy/project/reid_multiview/
# sh run.sh

#################################################################################
# Vis
#################################################################################
# cd /home/hy/project/reid_multiview/v3_vis
# sh run.sh

#################################################################################
# progress
#################################################################################
# # 构建多视角融合的baseline，仅用传播机制，重点改了resnet结构
# rm -rf /home/hy/project/reid_multiview/v3_backbone_P/results
# cd /home/hy/project/reid_multiview/v3_backbone_P
# sh run.sh

# resnet 第三层正交 
rm -rf /home/hy/project/reid_multiview/v4_P_ResL3Ort/results
cd /home/hy/project/reid_multiview/v4_P_ResL3Ort
sh run.sh
