import os
from parse import parse
import sys
from hq.src.utils import gather_data_gen, run_parser, get_algoname
from hq.src.eval_utils import write_eval_line
from hq.src.target_utils import to_onehot
from gm_hmm.src.utils import load_model, save_model

if __name__ == "__main__":
    this_algo = get_algoname(__file__)
    parser = run_parser(desc=this_algo)

    args = parser.parse_args()
    mdl_folder = args.mdl
    data_folder = args.data
    exp_folder = args.exp
    log_folder = args.log

    compute_type, isplit, feat_mode = parse("{}-split{}-feat{}", args.opt)
    if compute_type == "lc":
        datatype = "train"
    else:
        datatype = compute_type

    split_folder = os.path.join(exp_folder, "split{}".format(isplit))

    out_mdl = os.path.join(split_folder, mdl_folder, "{}.feat{}.mdl".format(this_algo,feat_mode))
    out_results = os.path.join(split_folder, log_folder, "{}.feat{}.report".format(this_algo, feat_mode))

    x_list, y_list = gather_data_gen(datatype, feat_mode,  os.path.join(split_folder, data_folder))

    if compute_type == "test":
        mdl = load_model(out_mdl)
        
        mdl.userdata["ttest"] = to_onehot(mdl.userdata["ttest"])
        mdl.userdata["ttrain"] = to_onehot(mdl.userdata["ttrain"])

        write_eval_line(mdl, out_mdl, out_results)

    sys.exit(0)

