#/bin/bash

model="neuralnets/mycnn/models/test_4kseq_model/"
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
seqlength=4000

echo ">>>CREATING MODEL<<<"
python neuralnets/src/cnn_structure.py \
	--model $model \
	--seqlength $seqlength

echo ">>>TRAINING MODEL<<<"
python neuralnets/src/train.py \
	--model $model \
	--posdirs ${pos_dirs[@]} \
	--negdirs ${neg_dirs[@]} \
	--trainreads 400000 \
	--valreads 5000 \
	--testreads 10 \
	--window $seqlength \
	--ratio 0.3 \
	--threshold 0.5 \
	--batchsize 32 \
	--gpulim 4096
