
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

# 构造backbone
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_6/results
cd /home/hy/project/reid_multiview/v4_IP_mv_6
sh run.sh

# v4_IP_mv_6, (共享-指定)损失 
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_7/results
cd /home/hy/project/reid_multiview/v4_IP_mv_7
sh run.sh

# v4_IP_mv_6, (共享-指定)损失/(共享)损失
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_8/results
cd /home/hy/project/reid_multiview/v4_IP_mv_8
sh run.sh


# v4_IP_mv_6, (共享-指定)损失/(共享)损失/(指定)损失
rm -rf /home/hy/project/reid_multiview/v4_IP_mv_9/results
cd /home/hy/project/reid_multiview/v4_IP_mv_9
sh run.sh
