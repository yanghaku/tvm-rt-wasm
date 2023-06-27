from model_info import get_ir_module_from_frontend
from module_process import build_ir_module, save_module
from utils import get_arg_parser, get_tvm_target

if __name__ == "__main__":
    opts = get_arg_parser().parse_args()

    ir_module, params = get_ir_module_from_frontend(opts)

    target = get_tvm_target(opts)

    m = build_ir_module(opts, ir_module, params, target)

    save_module(opts, m)

    print("build module success")
