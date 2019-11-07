MAKEFILES=makefiles
CONFIGS=configs
#models="dgen genhmm gmmhmm"
models_in=" gmmhmm"
ns_in="3 6"
niter_in="2 10 20"
nmix="2 4"
if [ "$1" == "print" ]
then
    echo "Models: " $models_in
    echo "Number of states (ns): " $ns_in
    echo "Number of iterations (niter): " $niter_in
    echo "Number of mixtures (nmix): " $nmix
    echo gmmhmm-ns"\{"${ns_in// /,}"\}"-niter"\{"${niter_in// /,}"\}"-nmix"\{"${nmix// /,}"\}"
exit 0
fi

for ns in $ns_in; do
for niter in $niter_in; do
for nnmix in $nmix; do
	for model in $models_in; do
		prefix=-ns$ns-niter$niter-nmix$nnmix
		cat $MAKEFILES/Makefile_$model | sed "s/$model/$model$prefix/g" > $MAKEFILES/Makefile_$model$prefix;
		cat $CONFIGS/$model.json | \
			sed -e "s/\"n_states\": 4/\"n_states\": $ns/g" \
			    -e "s/\"niter\": 2/\"niter\": $niter/g" \
			    -e "s/\"n_prob_components\": 2/\"n_prob_components\": $nnmix/g" > $CONFIGS/$model$prefix.json;

		ln -s -f run_$model.py bin/run_$model$prefix.py;
	done
done
done
done

