
###########################################################
# Visualize
###########################################################
mkdir -p results/outputs/models
wget -O results/outputs/models/model_119.pth "https://storage.googleapis.com/kaggle-script-versions/250223945/output/reid_multiview/main/results/occluded_duke/models/model_119.pth?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250714%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250714T090438Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=c024b1b89d293b2bc83e5e9276dcca4e07f97afde3e4de95c9ec364e04f39b4504b677d8b76f1889db699fc874974828bef0011aa11e2f8e0d76237d645b13d2eb3d4b3a73d18b5f45532846980257cd0b04e54648e3c14b8abeb06a4bcbb02a3d4fb6426b08b76b6b33cf62a97c087cc6dc148816763395686403ef11ca41805e0820d5e298d9f530584b5119714c4e0e7f74a5232501df0b2c70b48fa00d94c5aa7656b4400465367d85826bf87b06d282032988271831e9092d080293251d9d68db484b1a5011a43931d65152f564aa8b7621388ab1a31e37eb16accb93ea284e1d3dae160618c69aa0cd20e9e01228866bd306c74367b12c761aee945f2e"
python main.py --config_file "config/visualization.yml" 
tar -czf results.tar.gz results 
rm -rf results
