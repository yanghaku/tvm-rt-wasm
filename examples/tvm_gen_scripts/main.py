from module_process import build_module, save_module
from utils import get_arg_parser, get_tvm_target
from model_info import get_module_frontend

if __name__ == "__main__":
    opts = get_arg_parser().parse_args()

    mod, params = get_module_frontend(opts)

    target = get_tvm_target(opts)

    m = build_module(opts, mod, params, target)

    save_module(opts, m)

    print("build module success")
