#!/usr/bin/env sh
clear

dir=$(pwd)/

echo "Reading config..." >&2
. $dir"absa16.conf"

# ttd full path
pth=$dir$ttd/

# ---------------------------------------------------

ABSABaseAndEval ()
{

# create feature vector (Tree-LSTM , BOW , autoencoder)

echo -e "***** Create train vector for stage1 and train the model (using SVM) *****"
python preprocess_asp.py tr
python libsvm/python/train_asp.py $dom


# Stage 1 Predict

echo -e "***** Create test vector for stage1 and predict its category prob. distribution *****"
#produce te.svm.asp te.svm.pol
python preprocess_asp.py te
python libsvm/python/predict_asp.py $dom
#produce Out.asp


echo -e "***** Assigning categories using a threshold on the SVM prediction *****"
#produce teCln.PredAspTrg.xml
java -cp ./A.jar absa15.Do Assign $dom $dir $ttd $thr 1 "0"$pIdxArg

# Stage 2 Training

echo -e "***** Create train vector for stage2 and train the model (using SVM) *****"
python preprocess_pol.py tr
python libsvm/python/train_pol.py $dom

# Stage 2 Predict
echo -e "***** Creating test vectors for Stage2 and predict its polarity(Using gold aspect) *****"
python preprocess_pol.py te
python libsvm/python/predict_pol.py $dom

echo -e "***** Assigning polarities based on SVM prediction *****"
java -cp ./A_2015.jar absa15.Do Assign $dom $dir $ttd 0 2 "0"$pIdxArg

echo -e "\n"
echo -e "***** Evaluate Stage 1 Output (target and category) *****"

java -cp ./A.jar absa16.Do Eval -prd ${pth}"teCln.PrdAspTrg.xml"${suff} -gld ${pth}"teGld.xml"${suff} -evs 1 -phs A -sbt SB1


echo -e "***** Evaluate Stage 2 Output (Polarity) *****"
java -cp ./A.jar absa16.Do Eval ${pth}"teGldAspTrg.PrdPol.xml"${suff} ${pth}"teGld.xml"${suff} -evs 5 -phs B -sbt SB1

}

echo -e "*******************************************"
echo -e "Stage 1: Aspect and OTE extraction"
echo -e "Stage 2: Polarity classification"

echo -e "***** Validate Input XML *****"
java -cp ./A.jar absa16.Do Validate ${dir}${src} ${dir}"ABSA16.xsd" $dom

if [ "$xva" -eq 0 ]; then
	echo -e "***** Split Train Test *****"	
	java -cp ./A.jar absa16.Do Split $sfl $dir $ttd $src $fld $partIdx
	ABSABaseAndEval 
else 
	echo -e "***** Split *****" 	
	java -cp ./A.jar absa16.Do Split $sfl $dir $ttd $src $fld
	echo -e "\n***** Cross Validation*****\n"
	for i in $(eval echo {1..$fld}); do	  		  
	  echo -e "Round " $i	  	  	  
	  pIdxArg=" "$(($i-1))
	  suff="."$(($i-1))
	  echo $pIdxArg $suff
	  ABSABaseAndEval
	  echo -e "\n"
	done
fi
echo -e "*******************************************"


