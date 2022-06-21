from CP_MixedNet_attn import BRT
import torch
from torchtyping import patch_typeguard
from typeguard import typechecked


def test_recurrence():
    device = 'cuda:0'
    state_len = 40
    batch_size = 16
    channel_input = 8
    window = 100
    attn =  BRT(device=device,channel_p=21,channel_temp=32,conv_pers_window=7,window=100,input_channel=8,
                      dim_emb=8, dim_state=8, dim_h=8, state_len=state_len,
                      heads=4,batch_size=batch_size).to(device)
    inputs = [torch.randn((batch_size,channel_input,state_len,1,window)) for _ in range(3)]
# state_len and seq_len should be accord with each other.
# dim_state and dim should be accord with each other(in this case)
    for x in inputs:
        x = x.to(device)
        output = attn(x)
        print(output.shape)


