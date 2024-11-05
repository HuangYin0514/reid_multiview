
# ssh hy@10.244.37.93
# screen -D -r training
# conda activate py396
# cd /home/hy/project/reid_multiview/v4_3D_embedding
# sh run.sh

#################################################################################

rm -rf /home/hy/project/reid_multiview/v4_SEAM_3/results
cd /home/hy/project/reid_multiview/v4_SEAM_3
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_SEAM_4/results
cd /home/hy/project/reid_multiview/v4_SEAM_4
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_hierarchical_2/results
cd /home/hy/project/reid_multiview/v4_hierarchical_2
sh run.sh
