
# conda activate py396
# cd /home/hy/project/reid_multiview/v4_3D_embedding
# sh run.sh

#################################################################################


rm -rf /home/hy/project/reid_multiview/v4_Custom_Resnet_backbone/results
cd /home/hy/project/reid_multiview/v4_Custom_Resnet_backbone
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_CoNorm/results
cd /home/hy/project/reid_multiview/v4_CoNorm
sh run.sh

rm -rf /home/hy/project/reid_multiview/v4_SEAM/results
cd /home/hy/project/reid_multiview/v4_SEAM
sh run.sh

# rm -rf /home/hy/project/reid_multiview/v4_Attention_v2/results
# cd /home/hy/project/reid_multiview/v4_Attention_v2
# sh run.sh

