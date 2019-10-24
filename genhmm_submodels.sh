MAKEFILES=makefiles
CONFIGS=configs
#models="dgen genhmm gmmhmm"
models_in="dgenhmm genhmm"
ns_in="3 6 9"
niter_in="2 10 20 50"
nchain="4 8"
net_H="2"
nmix="1"
if [ "$1" == "print" ]
then
    echo "Models: " $models_in
    echo "Number of states (ns): " $ns_in
    echo "Number of iterations (niter): " $niter_in
    echo "Number of chains (nchain): " $nchain
    echo "Number of mixtures (nmix): " $nmix
    echo "Networks height (netH): " $net_H
    echo "\{"${models_in// /,}"\}"-ns"\{"${ns_in// /,}"\}"-niter"\{"${niter_in// /,}"\}"-nchain"\{"${nchain// /,}"\}"-nmix"\{"${nmix// /,}"\}"-netH"\{"${net_H// /,}"\}"
exit 0
fi

for ns in $ns_in; do
for niter in $niter_in; do
for nnchain in $nchain ; do
for netH in $net_H; do
for nnmix in $nmix; do
	for model in $models_in; do
		prefix=-ns$ns-niter$niter-nchain$nnchain-nmix$nnmix-netH$netH
		cat $MAKEFILES/Makefile_$model | sed "s/$model/$model$prefix/g" > $MAKEFILES/Makefile_$model$prefix;
		cat $CONFIGS/$model.json | \
			sed -e "s/\"n_states\": 4/\"n_states\": $ns/g" \
			    -e "s/\"niter\": 2/\"niter\": $niter/g" \
			    -e "s/\"net_nchain\": 4/\"net_nchain\": $nnchain/g" \
			    -e "s/\"net_H\": 64/\"net_H\": $netH/g" \
			    -e "s/\"n_prob_components\": 2/\"n_prob_components\": $nnmix/g" > $CONFIGS/$model$prefix.json;

		ln -s -f run_$model.py bin/run_$model$prefix.py;
	# 	mkdir exp/split{1..10}/models/dgen.dgenhmm-ns$ns.feat0;
	done
done
done
done
done
done

