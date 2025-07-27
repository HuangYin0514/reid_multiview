###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

###########################################################
# Runs
###########################################################
# python main.py --config_file "config/method.yml"  

# python main.py --config_file "config/occluded_reid/0.yml" 
# python main.py --config_file "config/occluded_reid/1.yml" 
# python main.py --config_file "config/occluded_reid/2.yml" 
# python main.py --config_file "config/occluded_reid/3.yml" 
# python main.py --config_file "config/occluded_reid/4.yml"
# python main.py --config_file "config/occluded_reid/5.yml"
# python main.py --config_file "config/occluded_reid/6.yml"
# python main.py --config_file "config/occluded_reid/7.yml"
# python main.py --config_file "config/occluded_reid/8.yml"
# python main.py --config_file "config/occluded_reid/9.yml"

###########################################################
# Visualize
###########################################################
mkdir -p results/outputs/models
wget -O results/outputs/models/model_114.pth "https://storage.googleapis.com/kaggle-script-versions/250844267/output/reid_multiview/main/results/outputs/models/model_114.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250727%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250727T161254Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=6cedfff84aedfd72ec487df4e50036e2a69b844fa96afbcfecfcafd8b4f5fd5504370b991a1874ac8650050cc3c624769196c2063d6a4f0b60ff8bdd55c804db53651e52e648885f67285a9e54ab3e086e9d120303efe6ca395958dc6c857f69104851e4c1b4669ee616d9ff484c2ccbcda6cd9bb2239a36ef65c8838965606a621f8847ed31d5a371e3e5c6a609496dd0e257d5f3de7278a04073053037af3a6736960021accb207c7204330afe1800430e41dad20e04e2a6033ad856dff1f0238c845ac50095d7db74f174e09cd20ff5210c66e73d5603f38a8452a023e18b68bc7c47c25903f76b273b0047a2dd3e5a9d2bdb21c93bc310d91fe8f8724c09"
python main.py --config_file "config/method.yml" TASK.MODE=visualization TEST.RESUME_TEST_MODEL=114 
# tar -czf results.tar.gz results
# rm -rf results/outputs/actmap results/outputs/rank results/outputs/tSNE 
