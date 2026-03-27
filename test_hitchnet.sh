MODEL_NAME=dummy_bev_resnet_regression_norm_aug1

python3 -m scripts.test \
  --config configs/experiments/${MODEL_NAME}.yaml \
  --ckpt ckpts/${MODEL_NAME}_seed44/best.pth \
  --num_workers 16 \
  --trailer_type charger 
  # --save_err_bev \
  # --err_bev_max 10 \
  # --save_best_bev \
  # --plot \
  # --plot_bins 5 