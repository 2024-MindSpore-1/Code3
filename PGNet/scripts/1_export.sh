python tools/export.py \
    --model_name_or_config configs/e2e/pgnet/pg_r50_totaltext.yaml \
    --local_ckpt_path work_dir/pgnet/best.ckpt \
    --data_shape 768 768 \
    --save_dir work_dir/pgnet/export
