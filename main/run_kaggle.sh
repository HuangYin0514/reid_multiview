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
wget -O results/outputs/models/model_114.pth "https://storage.googleapis.com/kaggle-script-versions/250000647/output/reid_multiview/main/results/occluded_duke/models/model_114.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250716%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250716T130236Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=0a996a9c2995bdf1519914a14b4229291db1bf775148d3c3416ee4e3ffcb7eef3e40833dac851be07879489fd0052e753a778bc956df07ec787e6ca22e0c8c867730eaa4b685070131c73d58d2dd8e797ae80051ff975366f9c36b72e4eb62d830ab1f393aa9ca36b131d2eb45ad65c4508854c3e3acf9e228cd29b5b1ea5af24148ae4371c603a7c62200bc16e248f5914860d649fa9b80400de0066ee5a6bf31dac87976e04d86b9c20e5a107e9dc7f1603f798787195bf6b2e587b518e735e9420384c97ea39dcc86bbaa6b489e04376e16cc20dccf5de30fc38b481b1a271e35ca9f32a141bdb816bb6460a0150aa785969cfe50172e064b5120dacc2fc0"
python main.py --config_file "config/visualization.yml" 
tar -czf results.tar.gz results
rm -rf results/outputs/actmap results/outputs/rank results/outputs/tSNE 
