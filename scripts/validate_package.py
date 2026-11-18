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
    assert {'README.md','ablation.md','代码框架逻辑.md','sources.md','overlap.md','Steering_OPD_完整研究方案.md'} <= {p.name for p in docs}
    md_files = [ROOT/'README.md',ROOT/'AGENTS.md',ROOT/'TODO.md',ROOT/'输出结构.md',*docs,
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
    config=read(ROOT/'configs/ablation.json')
    from steer_opd.compass_campaign import validate_config,make_base_job
    from steer_opd.compass_job import load_codecs
    from steer_opd.prefix_matching import METRIC_VERSION, RULE_VERSION, P_DEFINITION
    validate_config(config)
    assert config['experiment']=='ablation' and config['phase']=='overlap'
    from steer_opd.stage_io import layout_paths
    assert config['step_dirs']==layout_paths(config)
    assert config['run_dir']==config['step_dirs']['3']
    assert config['data_dir']==config['step_dirs']['0']
    assert config['input_data_dir']==config['step_dirs']['0']
    assert config['question_count']==1000 and config['confirm_top_n']==4
    assert config['execution_hold']=='awaiting_user_approval'
    assert config['require_dual_metrics'] and config['bind_checkpoint_content']
    assert config['request_seed_policy']=='constant_42' and config['reproducibility_protocol']==1
    assert config['retention_policy']=='minimal_v1' and config['retain_intermediates'] is False
    assert config['resume_granularity']=='engine_call'
    assert config['metric_version']==METRIC_VERSION and config['matching_rule_version']==RULE_VERSION
    assert {config[k] for k in ('screen_backend','extraction_backend','probability_backend')}=={'easysteer'}
    routes={
        'reference_and_source_generation':('scripts/generate_static_sources.py','engine.generate'),
        'shared_S0_activation_capture':('src/steer_opd/easysteer_extraction.py','hs.capture'),
        'native_teacher_scoring':('scripts/score_compass_sources.py','fixed_prefix_scores'),
        'baseline_and_four_fixed_schemes':('scripts/run_overlap_ablation.py','fixed_prefix_scores'),
        'fixed_prefix_engine_call':('src/steer_opd/easysteer_static.py','llm.generate')}
    routes['teacher_space_S_T_scoring']=('src/steer_opd/teacher_screen.py','fixed_prefix_scores')
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
        probability_statistics='vLLM raw Top16 IDs/logprobs only; exact matching on at most 32 candidate nodes; no full-vocabulary collector',
        cpu_work=['tokenization','byte/span alignment','mean pooling','T-S bank construction','metrics','serialization','plots'],
        hf_model_forward='Explicit hf_batched_static numerical reference only; no automatic fallback',
        resume_granularity='engine_call',gpu_validated=False)
    assert not any((ROOT/'src/steer_opd'/name).exists() for name in ('byte_probabilities.py','engine_projection.py'))
    codecs=load_codecs(config)
    job=make_base_job(config,codecs,'unused_rows.json')
    assert len(job['conditions'])==4 and len(job['banks'])==2
    from steer_opd.method_config import TEACHER_DIRECTION_VERSION
    assert config.get('include_teacher_ablation') is False
    assert config['teacher_layers']==[29] and config['teacher_direction_version']==TEACHER_DIRECTION_VERSION
    reverse={**config,'include_teacher_ablation':True,'confirm_top_n':8}
    validate_config(reverse)
    reverse_job=make_base_job(reverse,codecs,'unused_rows.json')
    assert len(reverse_job['conditions'])==8 and len(reverse_job['banks'])==4
    assert all(c['direction']=='S-T' and c['steer_role']=='teacher' and c['layers']==[29]
               for c in reverse_job['conditions'][4:])
    assert {c['strength_policy'] for c in job['conditions']}=={'fixed'}
    assert 'strength_schedule' not in job
    assert set(codecs)=={'student','teacher'}
    assert 'reference' not in config and 'reference_scores' not in job
    assert [codecs[r].support_size for r in ('student','teacher')]==[151669,248077]
    source=ROOT/config['question_source']
    base=ROOT/config['base_question_source']
    assert sha(base)=='c000d5875eb6944eaac179dc92a0ebd4abd0dfefc8021cdcff1230d1d200e7e4'
    questions=read(source);assert len(questions)==len({q['id'] for q in questions})==config['question_count']
    assert len({q['question_hash'] for q in questions})==len(questions)
    assert all(hashlib.sha256(q['question'].encode()).hexdigest()==q['question_hash'] for q in questions)
    assert all(q.get('domain','math')=='math' for q in questions)
    by_id={q['id']:q['question'] for q in questions}
    assert all(by_id.get(q['id'])==q['question'] for q in read(base))
    sampling=read(ROOT/config['data_dir']/'sampling_manifest.json')
    assert sampling['question_source_sha256']==sha(source) and sampling['base_source_sha256']==sha(base)
    assert sampling['questions']==1000 and sampling['preserved_base_questions']==100 and sampling['added_questions']==900
    from steer_opd.method_config import DIRECTION_VERSION
    assert config['direction']=='T-S' and config['direction_version']==DIRECTION_VERSION
    if (ROOT/config['data_dir']/'split.json').exists():
        split=read(ROOT/config['data_dir']/'split.json')
        assert len(split['select_ids'])==700 and len(split['confirm_ids'])==300
        assert not set(split['select_ids'])&set(split['confirm_ids'])
        assert set(split['select_ids'])|set(split['confirm_ids'])==set(by_id)
        assert split['source_sha256']==sha(source)
    files=[*(ROOT/'src/steer_opd').rglob('*.py'),*(ROOT/'scripts').glob('*.py')]
    for path in files:
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('steer_opd.'):
                module=ROOT/'src'/Path(*node.module.split('.'))
                assert module.with_suffix('.py').is_file() or (module/'__init__.py').is_file(),(path,node.module)
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.fullmatch(r'scripts/[a-z_]+\.(?:py|sh)',node.value):
                assert (ROOT/node.value).is_file(),(path,node.value)
    assert (ROOT/'src/steer_opd/easysteer_worker.py').is_file()  # Dynamic vLLM worker import.
    result={'status':'passed','paper_sources_verified':len(papers),'current_documents':len(docs),
            'local_links_checked':links,'python_sources_parsed':len(files),'training_question_candidates':len(questions),
            'schemes':4,'banks':2,'recorded_metrics':['overlap','P_overlap'],
            'optional_teacher_ablation':{'schemes':8,'banks':4,'teacher_layers':[29],
                'direction':'S-T','gpu_validated':False},
            'annealing':'deferred_to_future_training; no per-question overlap schedule',
            'experiment':config['experiment'],'phase':config['phase'],
            'direction':'T-S','direction_version':DIRECTION_VERSION,
            'metric_version':METRIC_VERSION,'matching_rule_version':RULE_VERSION,'P_overlap_definition':P_DEFINITION,
            'execution_hold':config['execution_hold'],'seed':42,'request_seed_policy':'constant_42','gpu_started':False,'parameter_updates':0,
            'resume_granularity':'engine_call','backend_audit':backend_audit,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    BACKEND_REPORT.parent.mkdir(parents=True,exist_ok=True)
    BACKEND_REPORT.write_text(json.dumps(backend_audit,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='source_sha256'})


if __name__=='__main__':main()
