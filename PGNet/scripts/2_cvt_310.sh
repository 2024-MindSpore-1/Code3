converter_lite \
    --fp16=on \
    --fmk=MINDIR \
    --modelFile=work_dir/pgnet/export/pg_r50_totaltext.mindir \
    --optimize=ascend_oriented \
    --outputFile=work_dir/pgnet/cvt_310_fp16 \
    --saveType=MINDIR
