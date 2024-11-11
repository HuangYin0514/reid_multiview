
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
# rm -rf /home/hy/project/reid_multiview/v4_LP_2l_2/results
# cd /home/hy/project/reid_multiview/v4_LP_2l_2
# sh run.sh

rm -rf /home/hy/project/reid_multiview/v3_P/results
cd /home/hy/project/reid_multiview/v3_P
sh run.sh

rm -rf /home/hy/project/reid_multiview/v3_LP_format/results
cd /home/hy/project/reid_multiview/v3_LP_format
sh run.sh

# 构建多视角融合的baseline，仅用传播机制，重点改了resnet结构
rm -rf /home/hy/project/reid_multiview/v3_P_format/results
cd /home/hy/project/reid_multiview/v3_P_format
sh run.sh
