"""Validate the current package offline, without loading GPU models."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'audit/package_validation.json'
BACKEND_REPORT = ROOT/'audit/coarse_resume_20261114/backend_audit.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    papers = []
    for manifest in ('papers/manifest.json','ref/manifest.json','ref/archive/manifest.json'):
        papers.extend(read(ROOT/manifest)['papers'])
    for paper in papers:
        for field in ('pdf','source_html','source_text'):
            path_key = 'pdf_file' if field=='pdf' else field
            path = ROOT/paper[path_key]
            assert sha(path)==paper[field+'_sha256'], path
            if field=='pdf': assert path.read_bytes().startswith(b'%PDF-'), path
        for locator in paper.get('evidence_locators',[]):
            assert locator['marker'].casefold() in (ROOT/paper['source_text']).read_text().casefold()
    for manifest in ('ref/easysteer_runtime/manifest.json','ref/implementation_sources/manifest.json'):
        for entry in read(ROOT/manifest)['files']:
            assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    metadata = ROOT/'assets/metadata/Qwen3.5-35B-A3B'
    for entry in read(metadata/'source_manifest.json')['files']:
        assert sha(metadata/entry['path'])==entry['sha256'],entry['path']
    docs = list((ROOT/'docs').glob('*.md'))
    assert {p.name for p in docs}=={'README.md','experiment.md','implementation.md','sources.md','Steering_OPD_完整研究方案.md'}
    md_files = [ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'TODO.md',*docs,
                ROOT/'ref/README.md',ROOT/'ref/archive/README.md']
    links=0
    for p in md_files:
        text=p.read_text();assert text.count('```')%2==0,p
        for url in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
            if url.startswith(('http:','https:','mailto:','#')):continue
            target=(p.parent/url.split('#')[0]).resolve()
            assert target.is_relative_to(ROOT.resolve()),(p,url)
            assert target.exists() or target in (REPORT.resolve(),BACKEND_REPORT.resolve()),(p,url)
            links+=1
    config=read(ROOT/'configs/static_dual_overlap.json')
    from steer_opd.compass_campaign import validate_config,make_base_job
    from steer_opd.compass_job import load_codecs
    validate_config(config)
    assert config['execution_hold']=='awaiting_user_approval'
    assert config['require_dual_metrics'] and config['bind_checkpoint_content']
    assert config['request_seed_policy']=='constant_42' and config['reproducibility_protocol']==1
    assert config['retain_intermediates']
    assert config['resume_granularity']=='engine_call'
    assert {config[k] for k in ('screen_backend','extraction_backend','probability_backend')}=={'easysteer'}
    routes={
        'reference_and_source_generation':('scripts/generate_static_sources.py','engine.generate'),
        'shared_S0_activation_capture':('src/steer_opd/easysteer_extraction.py','hs.capture'),
        'native_T_Ref_scoring':('scripts/score_compass_sources.py','fixed_prefix_scores'),
        'baseline_and_eight_schemes':('scripts/run_static_overlap_screen.py','fixed_prefix_scores'),
        'fixed_prefix_engine_call':('src/steer_opd/easysteer_static.py','llm.generate')}
    backend_audit={}
    for name,(file,entrypoint) in routes.items():
        tree=ast.parse((ROOT/file).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        matching=[n for n in calls if ast.unparse(n.func)==entrypoint]
        assert len(matching)==1,(file,entrypoint)
        assert not any(isinstance(n,ast.Name) and n.id=='RowCheckpoint' for n in ast.walk(tree)),file
        assert not any(isinstance(n,ast.Attribute) and n.attr in ('enqueue','step','_add_completion_requests') for n in ast.walk(tree)),file
        backend_audit[name]={'source':file,'line':matching[0].lineno,'entrypoint':entrypoint,'backend':'easysteer_vllm'}
    backend_audit.update(
        scope='Current zero-update panel and explicit HF acceptance/reference; legacy execution code removed',
        full_vocabulary_statistics='Custom worker PyTorch aggregation on existing vLLM logits; no HF model forward',
        cpu_work=['tokenization','byte/span alignment','mean pooling','bank construction','metrics','serialization','plots'],
        hf_model_forward='Explicit hf_batched_static numerical reference only; no automatic fallback',
        resume_granularity='engine_call',gpu_validated=False)
    codecs=load_codecs(config)
    job=make_base_job(config,codecs,'unused_rows.json')
    assert len(job['conditions'])==8 and len(job['banks'])==4
    assert [codecs[r].support_size for r in ('student','teacher','reference')]==[151669,248077,248077]
    source=ROOT/config['question_source']
    assert sha(source)=='c000d5875eb6944eaac179dc92a0ebd4abd0dfefc8021cdcff1230d1d200e7e4'
    questions=read(source);assert len(questions)==len({q['id'] for q in questions})==100
    if (ROOT/config['data_dir']/'split.json').exists():
        split=read(ROOT/config['data_dir']/'split.json')
        assert len(split['select_ids'])==70 and len(split['confirm_ids'])==30
        assert not set(split['select_ids'])&set(split['confirm_ids'])
    files=[*(ROOT/'src/steer_opd').glob('*.py'),*(ROOT/'scripts').glob('*.py')]
    for path in files:
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('steer_opd.'):
                assert (ROOT/'src'/Path(*node.module.split('.'))).with_suffix('.py').is_file(),(path,node.module)
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.fullmatch(r'scripts/[a-z_]+\.(?:py|sh)',node.value):
                assert (ROOT/node.value).is_file(),(path,node.value)
    assert (ROOT/'src/steer_opd/easysteer_worker.py').is_file()  # Dynamic vLLM worker import.
    result={'status':'passed','paper_sources_verified':len(papers),'current_documents':len(docs),
            'local_links_checked':links,'python_sources_parsed':len(files),'training_question_candidates':len(questions),
            'schemes':8,'banks':4,'recorded_metrics':['overlap','P_overlap'],
            'execution_hold':config['execution_hold'],'seed':42,'request_seed_policy':'constant_42','gpu_started':False,'parameter_updates':0,
            'resume_granularity':'engine_call','backend_audit':backend_audit,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    BACKEND_REPORT.parent.mkdir(parents=True,exist_ok=True)
    BACKEND_REPORT.write_text(json.dumps(backend_audit,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='source_sha256'})


if __name__=='__main__':main()
