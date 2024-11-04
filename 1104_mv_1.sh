VERSION=1

python feature_extract.py \
  --method netvlad \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front \
  --version $VERSION

python feature_extract.py \
  --method netvlad \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/front\
  --version $VERSION


python feature_extract.py \
  --method cosplace \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front\
  --version $VERSION

python feature_extract.py \
  --method cosplace \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/front\
  --version $VERSION


python feature_extract.py \
  --method mixvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front\
  --version $VERSION

python feature_extract.py \
  --method mixvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/front\
  --version $VERSION


python feature_extract.py \
  --method gem \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front\
  --version $VERSION

python feature_extract.py \
  --method gem \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/front\
  --version $VERSION


python feature_extract.py \
  --method convap \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front\
  --version $VERSION

python feature_extract.py \
  --method convap \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/front\
  --version $VERSION


python feature_extract.py \
  --method transvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0519/front\
  --version $VERSION

python feature_extract.py \
  --method transvpr \
  --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/0828/front\
  --version $VERSION


python feature_match.py\
  --version $VERSION

python eval/eval.py\
  --version $VERSION