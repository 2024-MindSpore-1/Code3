converter_lite \
    --configFile=scripts/quant/configs/fixed.txt \
    --fmk=MINDIR \
    --modelFile=work_dir/pgnet_fp16/export/pg_r50_totaltext_fp16.mindir \
    --optimize=ascend_oriented \
    --outputFile=work_dir/pgnet_fp16/quant/cvt_310_fixed \
    --saveType=MINDIR
