"""
Example memory allocator on LSTM
The LSTM construction code is copied from mxnet example.

See line 46 for setting mirror stage.
"""
import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import memonger

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    # For LSTM, we need to set mirror stage as integer,
    # the mirror stage of the same name will be considered as the same cut-line
    next_c._set_attr(mirror_stage=str(seqidx))
    next_h._set_attr(mirror_stage=str(seqidx))
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.,
                concat_decode=True, use_loss=False):
    """unrolled lstm network"""
    # initialize the parameter symbols
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    last_hidden = []
    for seqidx in range(seq_len):
        # embeding layer
        with mx.AttrScope(ctx_group='embed'):
            data = mx.sym.Variable("t%d_data" % seqidx)
            hidden = mx.sym.FullyConnected(data=data,
                                           weight=embed_weight,
                                           no_bias=True,
                                           num_hidden=num_embed,
                                           name="t%d_embed" % seqidx)
        # stack LSTM
        for i in range(num_lstm_layer):
            if i==0:
                dp=0.
            else:
                dp = dropout
            with mx.AttrScope(ctx_group='layer%d' % i):
                next_state = lstm(num_hidden, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i, dropout=dp)
                hidden = next_state.h
                last_states[i] = next_state

        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        last_hidden.append(hidden)

    out_prob = []
    if not concat_decode:
        for seqidx in range(seq_len):
            with mx.AttrScope(ctx_group='decode'):
                fc = mx.sym.FullyConnected(data=last_hidden[seqidx],
                                           weight=cls_weight,
                                           bias=cls_bias,
                                           num_hidden=num_label,
                                           name="t%d_cls" % seqidx)
                label = mx.sym.Variable("t%d_label" % seqidx)
                if use_loss:
                    sm = mx.sym.softmax_cross_entropy(fc, label, name="t%d_sm" % seqidx)
                else:
                    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name="t%d_sm" % seqidx)
                out_prob.append(sm)
    else:
        with mx.AttrScope(ctx_group='decode'):
            concat = mx.sym.Concat(*last_hidden, dim = 0)
            fc = mx.sym.FullyConnected(data=concat,
                                       weight=cls_weight,
                                       bias=cls_bias,
                                       num_hidden=num_label)
            label = mx.sym.Variable("label")
            if use_loss:
                sm = mx.sym.softmax_cross_entropy(fc, label, name="sm")
            else:
                sm = mx.sym.SoftmaxOutput(data=fc, label=label, name="sm")
            out_prob = [sm]

    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
                          h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
        last_states[i] = state

    unpack_c = [state.c for state in last_states]
    unpack_h = [state.h for state in last_states]
    list_all = out_prob + unpack_c + unpack_h
    return mx.sym.Group(list_all)


def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
        name.endswith("gamma") or name.endswith("beta")

def get_input_shapes(rnn_sym, batch_size, num_hidden, input_size, seq_len):
    arg_names = rnn_sym.list_arguments()
    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size, input_size)
        elif name == "label":
            input_shapes[name] = (batch_size * seq_len, )
        elif name.endswith("label"):
            input_shapes[name] = (batch_size, )
        else:
            pass
    return input_shapes


# Some configurations of LSTM

# Set concate decoder to use a single softmax will increase memory usage.
# But can get a bit faster speed.
concat_decode=False
# Set use loss means we use softmax_cross_entropy loss
# This can help reduce memory
use_loss=True

batch_size = 64
seq_len = 1000
num_hidden = 1024
num_embed = 1024
input_size = 50
num_lstm_layer = 4
num_label = 5000

net = lstm_unroll(
    num_lstm_layer=num_lstm_layer,
    seq_len=seq_len, input_size=input_size,
    num_hidden=num_hidden, num_embed=num_embed,
    num_label=num_label,
    concat_decode=concat_decode,
    use_loss=use_loss)

ishapes = get_input_shapes(net,
                           batch_size=batch_size,
                           num_hidden=num_hidden,
                           input_size=input_size,
                           seq_len=seq_len)

net_mem_planned = memonger.search_plan(net, **ishapes)
old_cost = memonger.get_cost(net, **ishapes)
new_cost = memonger.get_cost(net_mem_planned, **ishapes)

print('Old feature map cost=%d MB' % old_cost)
print('New feature map cost=%d MB' % new_cost)
# You can savely feed the net to the subsequent mxnet training script.


