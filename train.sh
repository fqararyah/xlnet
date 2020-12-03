# --record_info_dir=/home/nahmad/models/xlnet/data/1/tfrecords
#nvprof --track-memory-allocations off --unified-memory-profiling off --print-gpu-trace --csv --log-file ./durations.csv \
python3 train_gpu.py \
  --record_info_dir=/home/nahmad/models/xlnet/data_small/tfrecords \
  --train_batch_size=4 \
  --seq_len=512 \
  --reuse_len=256 \
  --mem_len=384 \
  --perm_size=256 \
  --n_layer=24 \
  --d_model=2048 \
  --d_embed=2048 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=4096 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --num_core_per_host=2 \