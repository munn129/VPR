DIMS="2048 1024 512 256 128"
IMAGES="concat front"
IMAGESIZES="1280 640 320"
METHOD=transvlad
DATES="0519 0828"

POSTFIX="without_mlp_mixer"

for image in $IMAGES
do
  for imagesize in $IMAGESIZES
  do
    for dim in $DIMS
    do
      VERSION="${imagesize}_${image}_${dim}_${POSTFIX}"
      echo "=================================================="
      echo "${METHOD} ${VERSION} ${date} is start at $(date +%H):$(date +%M):$(date +%S)"
      
      for date in $DATES
      do
        python feature_extract.py \
          --method $METHOD \
          --dataset_dir /media/moon/moon_ssd/moon_ubuntu/post_oxford/$date/$image \
          --save_dir /media/moon/T7\ Shield/dim_ex \
          --version $VERSION \
          --dim $dim \
          --image_size $imagesize 

      done

        echo "$VERSION feature matching start at $(date +%H):$(date +%M):$(date +%S)"
        python feature_match.py \
          --version $VERSION \
          --method $METHOD \
          --save_dir dim_ex_result


        echo "$VERSION eval start at $(date +%H):$(date +%M):$(date +%S)"
        python eval/eval.py \
          --version $VERSION \
          --method $METHOD


        echo "${VERSION} is done at $(date +%H):$(date +%M):$(date +%S)"

    done
  done
done

echo "=================================================="
echo "Done at $(date +%H):$(date +%M):$(date +%S)"