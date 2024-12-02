VERSION=vertical

python feature_extract.py \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/concat \
  --version $VERSION \
  --save_dir /media/moon/T7\ Shield/concatenated

python feature_extract.py \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/concat \
  --version $VERSION \
  --save_dir /media/moon/T7\ Shield/concatenated


python feature_match.py\
  --version $VERSION \
  --save_dir concantenated_result

python eval/eval.py\
  --version $VERSION