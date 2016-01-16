#!/usr/bin/env sh
clear

dir=$(pwd)/

echo "Reading config..." >&2
. $dir"absa16.conf"


# Marcos
svm_dir=src/libsvm/python

ABSABaseAndEval ()
{

if [ "$xva" -eq 0 ]; then
	pIdxArg=0
fi
# create feature vector (Tree-LSTM , BOW , autoencoder)

########################
### Stage 1 Training ###
########################
echo -e "***** Create train vector for stage1 and train the model (using SVM) *****" >&2

#produce tr.asp.dat tr.asp.label
scripts/preprocess/preprocess_asp.py $dom train $pIdxArg

#produce SVMmodel/lap.model
$svm_dir/train_asp.py $dom $pIdxArg bow

##########################
### Stage 1 Predicting ###
##########################
echo -e "***** Create test vector for stage1 and predict its category prob. distribution *****" >&2

#produce te.svm.asp te.svm.pol
scripts/preprocess/preprocess_asp.py $dom te $pIdxArg
$svm_dir/predict_asp.py $dom $pIdxArg $thr bow


echo -e "***** Produce result (xml format) in slot1  *****" >&2
scripts/submit/submit_asp.py $dom $pIdxArg



########################
### Stage 2 Training ###
########################
#echo -e "***** Create train vector for stage2 and train the model (using SVM) *****"
#scripts/preprocess/preprocess_pol.py $dom train $pIdxArg
#$svm_dir/train_pol.py $dom $pIdxArg
#$svm_dir/train_pol_old.py $dom $pIdxArg

##########################
### Stage 2 Predicting ###
##########################

#echo -e "***** Creating test vectors for Stage2 and predict its polarity(Using gold aspect) *****"
#scripts/preprocess/preprocess_pol.py $dom te $pIdxArg
#$svm_dir/predict_pol.py $dom $pIdxArg
#$svm_dir/predict_pol_old.py $dom $pIdxArg
#scripts/submit/submit_pol.py $dom $pIdxArg

echo -e "\n"

echo -e "***** Evaluate Stage 1 Output (target and category) *****"

java -cp ./A.jar absa16.Do Eval -prd "output/${dom}_asp.xml.${pIdxArg}" -gld "tmp_data/teGld.xml.${pIdxArg}" -evs 1 -phs A -sbt SB1


#echo -e "***** Evaluate Stage 2 Output (Polarity) *****"
#java -cp ./A.jar absa16.Do Eval -prd "output/${dom}_pol.xml.${pIdxArg}" -gld "tmp_data/teGld.xml.${pIdxArg}" -evs 5 -phs B -sbt SB1

}


echo -e "***** Validate Input XML *****"
java -cp ./A.jar absa16.Do Validate ${dir}${src} ${dir}"ABSA16.xsd" $dom

if [ "$xva" -eq 0 ]; then
	echo -e "***** Split Train Test *****"	
	java -cp ./A.jar absa16.Do Split $sfl $dir tmp_data $src $fld $partIdx
	ABSABaseAndEval
else
	echo -e "***** Split *****"
	java -cp ./A.jar absa16.Do Split $sfl $dir tmp_data $src $fld
	echo -e "\n***** Cross Validation*****\n"
	for i in $(eval echo {1..$fld}); do
	  echo -e "Round " $i
	  pIdxArg=$(($i-1))
	  suff="."$(($i-1))
	  ABSABaseAndEval
	  echo -e "\n"
	done
fi
echo -e "*******************************************"
