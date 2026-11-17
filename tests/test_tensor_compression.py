"""Lossless candidate storage retains exact scores and whole-call recovery."""
import pytest
import torch
from steer_opd.experiment_runtime import StageCheckpoint
from steer_opd.resumable_core import read_json


def test_gzip_roundtrip_preserves_all_tensor_bits_metadata_and_auto_resume(tmp_path):
    values={'ids':torch.tensor([[1,7,2]],dtype=torch.int32),'logps':torch.tensor([[-.1,-.2,-.7]]),
            'overlap':torch.tensor([.75,float('nan')],dtype=torch.float64),
            'native_mask':torch.tensor([True,False]),'diagnostics':{'raw':True,'units':[{'x':.12345}]}}
    StageCheckpoint(tmp_path,'identity',tensor_compression='gzip').save_tensors({'q':values})
    assert (tmp_path/'q.pt.gz').exists() and not (tmp_path/'q.pt').exists()
    packet=read_json(tmp_path/'complete.json');assert packet['tensor_compression']=='gzip'
    restored=StageCheckpoint(tmp_path,'identity').load_tensors(['q'])['q']
    for key in ('ids','logps','overlap','native_mask'):
        assert values[key].dtype==restored[key].dtype
        assert values[key].view(torch.uint8).equal(restored[key].view(torch.uint8))
    assert restored['diagnostics']==values['diagnostics']
    with pytest.raises(ValueError,match='identity changed'):
        StageCheckpoint(tmp_path,'identity',tensor_compression='none')


def test_compressed_cache_rejects_corruption_and_keeps_plain_format(tmp_path):
    compressed=tmp_path/'compressed';StageCheckpoint(compressed,'id',tensor_compression='gzip').save_tensors({'q':{'x':torch.ones(2)}})
    (compressed/'q.pt.gz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='changed'):StageCheckpoint(compressed,'id').load_tensors(['q'])
    plain=tmp_path/'plain';StageCheckpoint(plain,'id').save_tensors({'q':{'x':torch.ones(2)}})
    assert (plain/'q.pt').exists() and StageCheckpoint(plain,'id').load_tensors(['q'])['q']['x'].equal(torch.ones(2))
