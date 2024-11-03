python feature_extract.py \
  --method transvlad \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/icrca/0519

python feature_extract.py \
  --method transvlad \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/icrca/0828

python feature_match.py

python eval/eval.py