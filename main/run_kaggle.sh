###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

###########################################################
# Runs
###########################################################
# python main.py --config_file "config/method.yml" TASK.NOTES=v859_visualization TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl

# python main.py --config_file "config/method.yml" TASK.NOTES=v859_visualization TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl MODEL.MODULE=Baseline

###########################################################
# Visualize
###########################################################
mkdir -p results/outputs/models
wget -O results/outputs/models/model_114.pth "https://storage.googleapis.com/kaggle-script-versions/251889179/output/reid_multiview/main/results/outputs/models/model_114.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250728%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250728T022543Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=acceacce8026b3af8c0934a5f3d446081e23a9f6c2da376d3a9027286ef32eb50f77e0ad3bccc2d240724fe2425b899179be5fb4d443551fc71fb5bc84200b88a50c2dcf836c84c896832746fc75e9a10ab9480bdd6ba5e08c74e77216527b12727af0c2d232a2c4dfb726f091369ee1fc29780616a172ecd0e41ba73a9fbaa89cbb417c8dc8db52133ca16a47e979dd2284e907187f57f3b99c5b0278f7ce05e90d44a820966de3b3dbe59c40cc280da00649cc8dd19f07dbed7089d7a2840b6bf7d7a9989b101840567d11db9fa7d18a73fe8a2a136c37fcfccfd81c90e67b9133ce60396412ae1ba24d239dae5d712d093a66b9b8b634377af0026f21ede5"

# python main.py --config_file "config/method.yml"  TEST.RESUME_TEST_MODEL=114 
python main.py --config_file "config/method.yml" TASK.NAME=R_H2GP_S2GA_M2PQF_C2Mkl TASK.NOTES=v861_Baseline_visualization TASK.MODE=visualization MODEL.MODULE=Baseline

tar -czf results.tar.gz results
rm -rf results/outputs/actmap results/outputs/rank results/outputs/tSNE 
