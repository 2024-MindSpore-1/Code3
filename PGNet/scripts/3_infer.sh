python deploy/py_infer/infer.py \
    --input_images_dir data/total_text/test/rgb/ \
    --precision_mode fp16 \
    --e2e_model_path work_dir/pgnet/cvt_310_fp16.mindir \
    --e2e_model_name_or_config deploy/py_infer/src/configs/e2e/pg_r50_totaltext.yaml \
    --res_save_dir outputs/pgnet/ \
    --vis_pipeline_save_dir outputs/pgnet/ \
    --show_log True \
    --save_log_dir outputs/pgnet/
