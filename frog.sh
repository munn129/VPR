VERSION=14

python feature_extract.py \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat \
  --version $VERSION

python feature_extract.py \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat \
  --version $VERSION


python feature_match.py\
  --version $VERSION

python eval/eval.py\
  --version $VERSION