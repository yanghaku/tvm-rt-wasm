import os

import tvm
from tvm import relay, runtime, autotvm
from tvm.autotvm.tuner import XGBTuner
from tvm.ir.transform import PassContext
from tvm.relay.backend.executor_factory import ExecutorFactoryModule, AOTExecutorFactoryModule, \
    GraphExecutorFactoryModule
from tvm.contrib.graph_executor import GraphModule
from tvm.runtime.executor.aot_executor import AotModule

import utils


def save_module(opts, executor_factory_module):
    if not os.path.exists(opts.out_dir):
        os.mkdir(opts.out_dir)

    lib_src_path = os.path.join(opts.out_dir, "graph.ll")
    if opts.dso:
        lib_path = os.path.join(opts.out_dir, "graph.so")
    else:
        lib_path = os.path.join(opts.out_dir, "graph.tar")
    params_path = os.path.join(opts.out_dir, "graph.params")
    json_path = os.path.join(opts.out_dir, "graph.json")
    json_c_path = os.path.join(opts.out_dir, "graph.json.c")
    executor_factory_module.get_lib().export_library(lib_path)
    if opts.dso_only:
        return

    with open(params_path, "wb") as f:
        f.write(runtime.save_param_dict(executor_factory_module.get_params()))

    if opts.emit_llvm:
        with open(lib_src_path, "w") as f:
            f.write(executor_factory_module.get_lib().get_source())
    if opts.executor == "graph":
        json = executor_factory_module.get_graph_json()
        with open(json_path, "w") as f:
            f.write(json)
        utils.data2c(json, json_c_path, "graph_json")


def build_module(opts, mod, params, target):
    if opts.dso:
        syslib = False
    else:
        syslib = True
    executor = relay.backend.Executor(opts.executor)
    if opts.tune:
        tasks = autotvm.task.extract_from_program(mod, target=target, params=params)

        if not os.path.exists(opts.out_dir):
            os.mkdir(opts.out_dir)
        log_file = os.path.join(opts.out_dir, opts.tune_log_file)
        tune_module(opts, log_file, tasks)

        with autotvm.apply_history_best(log_file):
            with PassContext(opt_level=3):
                return relay.build(mod, target=target, params=params, executor=executor,
                                   runtime=tvm.relay.backend.Runtime("cpp", {"system-lib": syslib}))
    else:
        with PassContext(opt_level=3):
            return relay.build(mod, target=target, params=params, executor=executor,
                               runtime=tvm.relay.backend.Runtime("cpp", {"system-lib": syslib}))


def tune_module(opts, log_file, tasks):
    tuning_option = {
        "n_trial": opts.tune_n_trial,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["n_trial"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["n_trial"], prefix=prefix),
                autotvm.callback.log_to_file(log_file),
            ],
        )


def run_module(opts, m: ExecutorFactoryModule, input_dict):
    if isinstance(m, GraphExecutorFactoryModule):
        executor_create_func = m.module.get_function(m.libmod_name)
        executor = GraphModule(executor_create_func(tvm.device(opts.device_target, 0)))
    elif isinstance(m, AOTExecutorFactoryModule):
        executor_create_func = m.module.get_function(m.libmod_name)
        executor = AotModule(executor_create_func(tvm.device(opts.device_target, 0)))
    else:
        raise 'unsupported executor type: ' + str(type(m))

    for k, v in input_dict.items():
        executor.set_input(k, v)
    executor.run()
    return executor
