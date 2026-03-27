MODEL_NAME=charger_bev_resnet_regression_dann_dummy


python3 -m scripts.train_dann \
  --config configs/experiments/${MODEL_NAME}.yaml