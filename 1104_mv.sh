python feature_extract.py \
  --method netvlad \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat

python feature_extract.py \
  --method netvlad \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat


python feature_extract.py \
  --method cosplace \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat

python feature_extract.py \
  --method cosplace \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat


python feature_extract.py \
  --method mixvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat

python feature_extract.py \
  --method mixvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat


python feature_extract.py \
  --method gem \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat

python feature_extract.py \
  --method gem \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat


python feature_extract.py \
  --method convap \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat

python feature_extract.py \
  --method convap \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat


python feature_extract.py \
  --method transvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat

python feature_extract.py \
  --method transvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat


python feature_match.py

python eval/eval.py