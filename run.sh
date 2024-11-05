
# ssh hy@10.244.37.93
# screen -D -r training
# conda activate py396
# cd /home/hy/project/reid_multiview/v4_3D_embedding
# sh run.sh

#################################################################################

rm -rf /home/hy/project/reid_multiview/v4_SEAM_5/results
cd /home/hy/project/reid_multiview/v4_SEAM_5
sh run.sh
