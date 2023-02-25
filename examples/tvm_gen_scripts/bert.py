# https://huggingface.co/docs/transformers/main/en/torchscript
import numpy as np
import torch
import tvm
from tvm import relay

from module_process import build_module, run_module
from utils import get_arg_parser, get_tvm_target


def get_bert_input_dummy(bert_name: str):
    from transformers import BertTokenizer

    enc = BertTokenizer.from_pretrained(bert_name)
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)
    masked_index = 8
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensors


def get_traced_model(bert_name, inputs):
    from transformers import BertModel

    bert_model = BertModel.from_pretrained(bert_name, torchscript=True)
    bert_model.eval()
    for p in bert_model.parameters():
        p.requires_grad_(False)
    traced_model = torch.jit.trace(bert_model, inputs)
    traced_model.eval()
    for p in traced_model.parameters():
        p.requires_grad_(False)
    return traced_model


def get_bert_frontend(bert_name: str):
    input_dummy = get_bert_input_dummy(bert_name)
    traced_model = get_traced_model(bert_name, input_dummy)

    shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in list(traced_model.graph.inputs())[1:]]
    mod_bert, params_bert = relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
    return mod_bert, params_bert


def validate_bert_model(opts):
    input_dummy = get_bert_input_dummy(opts.model)
    traced_model = get_traced_model(opts.model, input_dummy)

    # build tvm module
    shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in list(traced_model.graph.inputs())[1:]]
    mod_bert, params_bert = relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
    target = get_tvm_target(opts)
    m = build_module(opts, mod_bert, params_bert, target)
    # build input for tvm and run
    inputs = {
        'input_ids': tvm.runtime.ndarray.array(input_dummy[0].numpy()),
        'attention_mask': tvm.runtime.ndarray.array(input_dummy[1].numpy())
    }
    executor = run_module(opts, m, inputs)
    output0 = executor.get_output(0).numpy()
    output1 = executor.get_output(1).numpy()

    torch_output0, torch_output1 = (i.numpy() for i in traced_model(*input_dummy))
    print("output0.shape: ", torch_output0.shape, output0.shape)  # (1, 14, 1024) (1, 14, 1024)
    print("output1.shape: ", torch_output1.shape, output1.shape)  # (1, 1024) (1, 1024)
    print("max difference in output0: ", np.max(np.abs((output0 - torch_output0))))
    print("max difference in output1: ", np.max(np.abs((output1 - torch_output1))))


if __name__ == '__main__':
    parser = get_arg_parser()
    parser.add_argument("--validate", type=bool, default=False, help="validate the model")
    opt = parser.parse_args()

    if opt.validate:
        # validate model
        opt.host_target = 'native'
        validate_bert_model(opt)
    else:
        # just generate the inputs for classification
        dummy_input = get_bert_input_dummy(opt.model)
        # the input type is int64
        (input_ids, attention_mask) = (i.numpy().astype(np.int64) for i in dummy_input)
        print("input_ids: ", input_ids.shape, input_ids)
        print("attention_mask: ", attention_mask.shape, attention_mask)

        import os

        # with open(os.path.join(opt.out_dir, "inputs_ids.bin"), "wb") as f:
        #     f.write(input_ids.tobytes())
        # with open(os.path.join(opt.out_dir, "attention_mask.bin"), "wb") as f:
        #     f.write(attention_mask.tobytes())
        with open(os.path.join(opt.out_dir, "bert_mix_input.bin"), "wb") as f:
            f.write(input_ids.tobytes())
            f.write(attention_mask.tobytes())
