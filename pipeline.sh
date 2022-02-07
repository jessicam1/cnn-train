#/bin/bash

model="neuralnets/mycnn/models/test_4kseq_model/"
mkdir -p "$model"
logsdir="neuralnets/mycnn/tblogs/"
mkdir -p "$logsdir"
seqlength=4000
batchsize=32
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
experiments=(
	"K562_5EU_0_unlabeled_run"
	"K562_5EU_0_unlabeled_II_run"
	"K562_5EU_0_unlabeled_III_run"
	"K562_5EU_1440_labeled_run"
	"K562_5EU_1440_labeled_II_run"
	"K562_5EU_1440_labeled_III_run"
	"K562_5EU_60_labeled_heat_run"
	"K562_5EU_60_labeled_heat_II_run"
	"K562_5EU_60_labeled_heat_III_run"
	"K562_5EU_60_labeled_heat_IV_run"
	"K562_5EU_60_labeled_heat_V_run"
	"K562_5EU_60_labeled_run"
	"K562_5EU_60_labeled_II_run"
	"K562_5EU_60_labeled_III_run"
	"K562_5EU_60_labeled_IV_run"
	"K562_5EU_60_labeled_V_run"
	"K562_5EU_60_labeled_VI_run"
	)
predictionspath="/home/martinjl2/projects/ontml/analysis/predictions/results/$model"
mkdir -p $predictionspath
pospreds="pospreds.txt"

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
	--batchsize $batchsize \
	--gpulim 4096

echo ">>> GETTING PREDICTIONS FROM MODEL<<<" 
echo -e "library\tpositive_predictions\ttotal_predictions" > $predictionspath/$pospreds

for lib in "${experiments[@]}"
do
	echo ">>> MAKING PREDICTIONS DIRECTORY <<<"
	mkdir -p $predictionspath/$lib
	echo $predictionspath/$lib
	echo ">>> GETTING PREDICTIONS FOR $lib <<<"
	savefile="predictions.tab"
	pospreds="pospreds.txt"
	echo "predictions saved at $predictionspath/$savefile"
	python neuralnets/src/gen_predictions.py \
		--model $model \
		--library $DATA_DIR/$lib/guppy/ \
		--window $seqlength \
		--batchsize $batchsize \
		--threshold 0.5 \
		--gpulim 4096 \
		--verbose \
	| grep -v 'foo' \
	> $predictionspath/$lib/$savefile
	
	echo ">>> SUMMING POSITIVE PREDICTIONS FOR $lib <<<"
	echo -en "$lib \t" >> $predictionspath/$pospreds
	awk '{s+=$3}END{printf s "\t"}' $predictionspath/$lib/$savefile >> $predictionspath/$pospreds
	echo "predictions summed in output file."
	wc -l < $predictionspath/$lib/$savefile | awk '{print $1 -1}' >> $predictionspath/$pospreds 

done
wait
