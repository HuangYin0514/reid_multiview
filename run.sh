
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

rm -rf /home/hy/project/reid_multiview/v4_P/results
cd /home/hy/project/reid_multiview/v4_P
sh run.sh

# rm -rf /home/hy/project/reid_multiview/v4_LP_test/results
# cd /home/hy/project/reid_multiview/v4_LP_test
# sh run.sh
