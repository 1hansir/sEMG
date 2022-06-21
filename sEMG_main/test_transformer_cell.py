import sys
sys.path.append("..")

from block_recurrent_transformer.transformer_ad import *
import torch
from torchtyping import patch_typeguard


'''
@typechecked
def test_block_recurrent_attention():
    attn = BlockRecurrentAttention(128, 128, state_len=128)
    x = torch.randn((64, 128, 128))
    state = torch.randn((64, 128, 128))

    out, new_state = attn(x, state)
    assert out.shape == new_state.shape
'''


def test_recurrence():
    device = 'cuda:0'
    state_len = 30
    batch_size = 32
    channel_input = 8
    window = 100
    dim_state = 8
    dim_target = 8
    attn = make_model(device=device,N=5,
                      dim_emb=8, dim_state=dim_state, dim_h=8, state_len=state_len,
                      h=4,channel_input=channel_input,dim_target=dim_target).to(device)
    inputs = [torch.randn((batch_size,channel_input,state_len,1,window)).to(device) for _ in range(50)]
    state_src = torch.zeros(batch_size, state_len, dim_state).to(device)
    state_tgt = torch.randn(batch_size, state_len, dim_state).to(device)
# state_len and seq_len should be accord with each other.
# dim_state and dim should be accord with each other(in this case)

    for x in inputs:
        shape_src = state_src.shape
        shape_tgt = state_src.shape
        output,state_tgt,state_src = attn(src=x,state_src=state_src,tgt=x,state_tgt=state_tgt)
        assert shape_src == state_src.shape
        assert shape_tgt == state_tgt.shape
        assert output.shape[-1] == dim_target

