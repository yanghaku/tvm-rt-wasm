import os

import tvm
from tvm import relay, runtime, autotvm
from tvm.ir.transform import PassContext
from tvm.relay.backend.executor_factory import AOTExecutorFactoryModule, GraphExecutorFactoryModule

import utils


def _save_graph_executor_module(opts, module: GraphExecutorFactoryModule):
    lib_src_path = os.path.join(opts.out_dir, "graph.ll")
    if opts.dso:
        lib_path = os.path.join(opts.out_dir, "graph.so")
    else:
        lib_path = os.path.join(opts.out_dir, "graph.tar")
    params_path = os.path.join(opts.out_dir, "graph.params")
    json_path = os.path.join(opts.out_dir, "graph.json")
    json_c_path = os.path.join(opts.out_dir, "graph.json.c")
    module.get_lib().export_library(lib_path)
    if opts.dso_only:
        return

    with open(params_path, "wb") as f:
        f.write(runtime.save_param_dict(module.get_params()))

    json = module.get_graph_json()
    with open(json_path, "w") as f:
        f.write(json)
    utils.data2c(json, json_c_path, "graph_json")

    if opts.emit_llvm:
        with open(lib_src_path, "w") as f:
            f.write(module.get_lib().get_source())


def save_module(opts, module):
    """Save the module built by relay or relax
    """
    if not os.path.exists(opts.out_dir):
        os.mkdir(opts.out_dir)

    if isinstance(module, GraphExecutorFactoryModule):
        return _save_graph_executor_module(opts, module)
    elif isinstance(module, AOTExecutorFactoryModule):
        if opts.dso:
            lib_path = os.path.join(opts.out_dir, "aot.so")
        else:
            lib_path = os.path.join(opts.out_dir, "aot.tar")
        module.get_lib().export_library(lib_path)
    elif isinstance(module, runtime.vm.Executable):  # relay vm
        if opts.dso:
            lib_path = os.path.join(opts.out_dir, "relay_vm.so")
        else:
            lib_path = os.path.join(opts.out_dir, "relay_vm.tar")
        code, lib = module.save()
        lib.export_library(lib_path)
        if opts.dso_only:
            return
        with open("relay_vm.ro", "wb") as f:
            f.write(code)
    elif isinstance(module, tvm.relax.Executable):  # relax vm
        if opts.dso:
            lib_path = os.path.join(opts.out_dir, "relax_vm.so")
        else:
            lib_path = os.path.join(opts.out_dir, "relax_vm.tar")
        module.export_library(lib_path)
    else:
        raise Exception('unsupported module type: ' + str(type(module)))


def run_module(opts, module, input_dict):
    """Run the module built by relay or relax
    """
    from tvm.contrib.graph_executor import GraphModule
    from tvm.runtime.executor.aot_executor import AotModule

    def _run_executor(_executor, _input_dict):
        for _k, _v in _input_dict.items():
            _executor.set_input(_k, _v)
        _executor.run()
        _num_outputs = _executor.get_num_outputs()
        return [_executor.get_output(_i) for _i in range(_num_outputs)]

    dev = tvm.device(opts.device_target, 0)
    if isinstance(module, GraphExecutorFactoryModule):
        executor_create_func = module.module.get_function(module.libmod_name)
        executor = GraphModule(executor_create_func(dev))
        return _run_executor(executor, input_dict)
    elif isinstance(module, AOTExecutorFactoryModule):
        executor_create_func = module.module.get_function(module.libmod_name)
        executor = AotModule(executor_create_func(dev))
        return _run_executor(executor, input_dict)
    elif isinstance(module, runtime.vm.Executable):
        # relay vm
        vm = runtime.vm.VirtualMachine(module, dev)
        return vm.run(**input_dict).numpy()
    elif isinstance(module, tvm.relax.Executable):
        from tvm import relax
        vm = relax.VirtualMachine(module, dev)
        main_func_name = "main"
        vm.set_input(main_func_name, **input_dict)
        vm.invoke_stateful(main_func_name)
        return vm.get_outputs(main_func_name).numpy()
    else:
        raise Exception('unsupported module type: ' + str(type(module)))


def build_ir_module(opts, ir_module, params, target):
    if opts.executor == "relax_vm":
        from tvm import relax
        with PassContext(opt_level=3):
            return relax.build(ir_module, target, params=params)
    elif opts.executor == "relay_vm":
        with PassContext(opt_level=3):
            return relay.vm.compile(ir_module, target, target_host=target, params=params)
    elif opts.executor != "graph" and opts.executor != "aot":
        raise Exception('unsupported backend type: ' + opts.executor)

    # graph or aot
    if opts.dso:
        sys_lib = False
    else:
        sys_lib = True
    executor = relay.backend.Executor(opts.executor)
    if opts.tune:
        tasks = autotvm.task.extract_from_program(ir_module, target=target, params=params)

        if not os.path.exists(opts.out_dir):
            os.mkdir(opts.out_dir)
        log_file = os.path.join(opts.out_dir, opts.tune_log_file)
        _tune_ir_module(opts, log_file, tasks)

        with autotvm.apply_history_best(log_file):
            with PassContext(opt_level=3):
                return relay.build(ir_module, target=target, params=params, executor=executor,
                                   runtime=relay.backend.Runtime("cpp", {"system-lib": sys_lib}))
    else:
        with PassContext(opt_level=3):
            return relay.build(ir_module, target=target, params=params, executor=executor,
                               runtime=relay.backend.Runtime("cpp", {"system-lib": sys_lib}))


def _tune_ir_module(opts, log_file, tasks):
    from tvm.autotvm.tuner import XGBTuner

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
