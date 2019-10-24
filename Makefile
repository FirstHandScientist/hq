SHELL=/bin/bash
PYTHON=$(abspath pyenv/bin/python)
BIN=bin

EXP=exp
MODELS=models
DATA=data
sLOG=log
gm_hmm=$(realpath src/gm_hmm)
LOG=log

ifndef
	proc_data=../../data/serialized/101/extrapolated/study_0/ml_6
endif

ifndef nsplits
	nsplits=$(shell echo $(proc_data)/split* | wc -w)
endif

ifndef splits
	splits={1..$(nsplits)}
endif

ifndef model
	model=logreg
endif

ifndef feats
	feats={1,2}
endif

MAKEFILES=makefiles
flags=.flags

inits=$(shell echo init-$(splits))
targets=$(shell echo $(flags)/{train,test}-split$(splits)-feat$(feats).$(model))

all: run

test:
	echo $(splits)
	echo $(nsplits)

help:
	@echo "Process pipeline for deep-news project."
	@echo "Targets:"
	@echo -e "help, prints this message."
	@echo -e "run, run the analysis with specified options."
	@echo "Options:"
	@echo -e "\tproc_data=<Processed patient data location.>\tdefault: proc_data=$(proc_data)"
	@echo -e "\tnsplits=<Number of split folders to use>\tdefault: count folders proc_data/split*"
	@echo -e "\tmodel=<logreg|elm|mlp|svm>\tdefault:model=$(model)"
	@echo -e "\tfeats=<feature type to use, see src/feat_utils.py>\tdefault: feats=$(feats)"

	
print:
	cat $(EXP)/$(LOG)/*.Acc

run: post_proc_results

init: $(inits)

init-%:
	mkdir -p $(EXP)/split$*/{data,log,models}
	cp $(proc_data)/split$*/* $(EXP)/split$*/data/


post_proc_results: $(targets)
	$(PYTHON) $(BIN)/acc_group.py -lines `echo $(EXP)/split$(splits)/log/*$(model)*feat$(feats).report` -log $(LOG) -exp $(EXP)
	#$(PYTHON) $(BIN)/lc_group.py -lines `echo $(EXP)/split$(splits)/log/*$(model)*feat$(feats).lc` -log $(LOG) -exp $(EXP)
	cat exp/log/$(model).feat*.Acc 
	

# Makefile of all available algorithms
include $(MAKEFILES)/Makefile_gmmhmm*
include $(MAKEFILES)/Makefile_genhmm*
include $(MAKEFILES)/Makefile_dgenhmm*


# $(flags)/%.class:  $(flags)/%.feats
$(flags)/%.class:
	$(PYTHON) $(BIN)/prepare_class.py -opt $* -exp $(EXP) -data $(DATA)
	touch $@


$(flags)/test-%.feats: $(flags)/train-%.feats
	$(PYTHON) $(BIN)/extract_features.py -opt test-$* -exp $(EXP) -data $(DATA)
	touch $@

$(flags)/train-%.feats:
	$(PYTHON) $(BIN)/extract_features.py -opt train-$* -exp $(EXP) -data $(DATA)
	touch $@


#Makefile_%:
#	@if [ $@ = Makefile_logreg ]; then\
#		echo "";\
#	else\
#		cat Makefile_logreg | sed 's/logreg/$*/g' > $@;\
#	fi

clean: clean-flags clean-data clean-models clean-log
	rm -f log*.log

clean-flags: clean-feats-flags clean-class-flags
	rm -f $(flags)/*-split*-feat*.*

clean-log:
	rm -rf $(EXP)/**/$(sLOG)/*
	rm -f $(EXP)/$(LOG)/*.{Acc,lc.pdf}

clean-models:
	rm -rf $(EXP)/**/$(MODELS)/*

clean-%-flags:
	rm -f $(flags)/*.$*

clean-data:
	rm -f $(EXP)/**/$(DATA)/{train,test}.feat*{_class*,}.pkl
	
.PHONY: results 	

.PRECIOUS: %.class %.feats

.SECONDARY:
