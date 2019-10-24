MAKEFILES=makefiles
CONFIGS=configs
models_in="logreg"
alphas="-5 -4 -3 -2 -1 0 1 2 3 4 5"
if [ "$1" == "print" ]
then
    echo "Models: " $models_in
    echo "Networks height (netH): " $alpha
    echo "\{"${models_in// /,}"\}"-alpha"\{"${alpha// /,}"\}"  
exit 0
fi

for alpha in $alphas; do
	for model in $models_in; do
		prefix=-alpha$alpha   
		cat $MAKEFILES/Makefile_$model | sed "s/$model/$model$prefix/g" > $MAKEFILES/Makefile_$model$prefix;
		cat $CONFIGS/$model.json | \
			sed  -e "s/\"alpha\": 0/\"alpha\": $alpha/g" > $CONFIGS/$model$prefix.json;

		ln -s -f run_$model.py bin/run_$model$prefix.py;
	# 	mkdir exp/split{1..10}/models/dgen.dgenhmm-ns$ns.feat0;
	done
done
