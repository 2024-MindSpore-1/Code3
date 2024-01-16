python tools/eval.py \
    -c configs/e2e/pgnet/pg_r50_totaltext.yaml \
    -o eval.ckpt_load_path=work_dir/pgnet_npu/best.ckpt \
        system.distribute=False
