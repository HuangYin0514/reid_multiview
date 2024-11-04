
# ssh hy@10.244.37.93
# screen -D -r training
# conda activate py396
# cd /home/hy/project/reid_multiview/v4_3D_embedding
# sh run.sh

#################################################################################

rm -rf /home/hy/project/reid_multiview/v3_backbone/results
cd /home/hy/project/reid_multiview/v3_backbone
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_SEAM/results
cd /home/hy/project/reid_multiview/v4_SEAM
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_SEAM_2/results
cd /home/hy/project/reid_multiview/v4_SEAM_2
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_hierarchical/results
cd /home/hy/project/reid_multiview/v4_hierarchical
sh run.sh


