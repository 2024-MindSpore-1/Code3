python tools/infer/text/predict_e2e.py \
    --image_dir data/total_text/test/rgb/img200.jpg \
    --e2e_algorithm PG \
    --e2e_amp_level O2 \
    --e2e_model_config configs/e2e/pgnet/pg_r50_totaltext.yaml \
    --e2e_model_dir work_dir/pgnet_mix_fp16_60_6/best.ckpt \
    --visualize_output True \
    --draw_img_save_dir outputs/pgnet
