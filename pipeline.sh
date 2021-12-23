#/bin/bash

model="neuralnets/mycnn/models/test07skew" 
mkdir -p "$model"
logsdir="neuralnets/mycnn/tblogs/"
mkdir -p "$logsdir"
pos_dirs=(
	"exp/K562_5EU_1440_labeled_run/guppy/"
	"exp/K562_5EU_1440_labeled_II_run/guppy/"
	"exp/K562_5EU_1440_labeled_III_run/guppy/"
)
neg_dirs=(
	"exp/K562_5EU_0_unlabeled_run/guppy/"
	"exp/K562_5EU_0_unlabeled_II_run/guppy/"
	"exp/K562_5EU_0_unlabeled_III_run/guppy/"
)


echo ">>>CREATING MODEL<<<"
python neuralnets/mycnn/src/cnn_structure.py -m $model

echo ">>>TRAINING MODEL<<<"
python neuralnets/src/train.py \
	--model $model \
	--posdirs $pos_dirs \
	--negdirs $neg_dirs \
	--trainreads 50000 \
	--valreads 10000 \
	--ratio 0.7 \
	--threshold 0.5 \
	--batchsize 32 \
	--gpulim 4096 \

