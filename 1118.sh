VERSION=640f
METHODS="netvlad cosplace mixvpr gem convap transvpr"
DATES="0519 0828"
IMAGE=front

for method in $METHODS
do
  for date in $DATES
  do
    echo "$date, $method is extracting"
    echo "date now : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)"
    python feature_extract.py \
      --method $method \
      --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/$date/$IMAGE \
      --save_dir /media/moon/T7\ Shield/ablation_study_with_image_size \
      --batch_size 1 \
      --version $VERSION
  done
done

echo "$VERSION feature matching start"
echo "date now : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)"
python feature_match.py \
--version $VERSION

echo "$VERSION eval start"
echo "date now : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)"
python eval/eval.py \
--version $VERSION