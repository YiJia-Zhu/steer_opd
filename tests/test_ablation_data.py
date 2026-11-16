"""Local data expansion preserves the old pool and never loads a GPU model."""
import importlib
from types import SimpleNamespace as NS

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steer_opd.resumable_core import ROOT


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('prepare_ablation_data')
    monkeypatch.setattr(module,'prompt_ids',lambda tokenizer,text,thinking:list(range(len(text)+tokenizer.extra)))
    return module


def test_expansion_preserves_base_and_filters_duplicates_domain_and_both_prompt_lengths(preparation):
    codecs={'student':NS(tokenizer=NS(extra=0)),'teacher':NS(tokenizer=NS(extra=4))}
    base=[{'id':'old','question':'base','answer':'0','teacher_prompt_ids':[999]}]
    candidates=[{'id':'dup','question':'base'}, {'id':'code','question':'code','domain':'code'},
                {'id':'too_long','question':'1234567'}, {'id':'new1','question':'one'}, {'id':'new2','question':'two'}]
    selected,rejected=preparation.select_expansion(base,candidates,3,codecs,10,False)
    assert [r['id'] for r in selected]==['old','new1','new2']
    assert selected[0]['question']==base[0]['question'] and 'teacher_prompt_ids' not in selected[0]
    assert rejected=={'duplicate':1,'non_math':1,'prompt_too_long':1}
    assert len({r['question_hash'] for r in selected})==3


def test_invalid_or_insufficient_pool_fails_without_silently_dropping_originals(preparation):
    codecs={'student':NS(tokenizer=NS(extra=0))}
    with pytest.raises(ValueError,match='cannot drop'):
        preparation.select_expansion([{'id':'a','question':'a'}],[],0,codecs,10,False)
    with pytest.raises(ValueError,match='Original frozen'):
        preparation.select_expansion([{'id':'a','question':'a'*11}],[],1,codecs,10,False)
    with pytest.raises(ValueError,match='Only 1'):
        preparation.select_expansion([{'id':'a','question':'a'}],[],2,codecs,10,False)


def test_local_sampler_is_seeded_math_only_and_records_exact_source_rows(preparation,tmp_path):
    def row(domain,text):
        return {'domain':domain,'source':'local','difficulty':3,
                'conversations':[{'from':'human','value':text},{'from':'gpt','value':'reference'}]}
    pq.write_table(pa.Table.from_pylist([row('code','code0'),row('math','math1'),row('math','math2')]),
                   tmp_path/'a.parquet',row_group_size=2)
    pq.write_table(pa.Table.from_pylist([row('science','science3'),row('math','math4')]),
                   tmp_path/'b.parquet',row_group_size=1)
    first,meta=preparation.local_candidates(tmp_path,3,42)
    second,_=preparation.local_candidates(tmp_path,3,42)
    assert first==second and meta['math_rows']==3
    assert {r['question']:r['source_global_row'] for r in first}=={'math1':1,'math2':2,'math4':4}
    assert all(r['domain']=='math' for r in first)
    assert len({r['id'] for r in first})==3
    assert len(meta['shards'])==2 and all(len(x['sha256'])==64 for x in meta['shards'])
