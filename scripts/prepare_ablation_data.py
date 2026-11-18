#!/usr/bin/env python3
"""Expand the frozen question pool from existing local OT3 shards; CPU only."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
from pathlib import Path
import random

from steer_opd.compass_campaign import frozen_json, prepare_questions
from steer_opd.compass_job import load_codecs, project_path
from steer_opd.direction_bank import file_sha256
from steer_opd.experiment_runtime import timestamp
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import ROOT, read_json
from steer_opd.stage_io import configure_layout


def question_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def select_expansion(base, candidates, count, codecs, max_prompt_tokens, enable_thinking):
    """Preserve every base question, then accept distinct prompt-valid math rows."""
    if count < len(base):
        raise ValueError('Expansion cannot drop the original question pool')
    selected=[];seen_ids=set();seen_questions=set();rejected=defaultdict(int)
    for origin, rows in (('base',base),('candidate',candidates)):
        for row in rows:
            if origin=='candidate' and len(selected)==count:break
            text=row['question'];sha=question_hash(text)
            reason=('non_math' if row.get('domain','math')!='math' else
                    'empty' if not text.strip() else
                    'duplicate' if sha in seen_questions or row['id'] in seen_ids else
                    'prompt_too_long' if any(len(prompt_ids(c.tokenizer,text,enable_thinking))>max_prompt_tokens
                                             for c in codecs.values()) else None)
            if reason:
                if origin=='base':raise ValueError(f'Original frozen question is invalid: {row["id"]}: {reason}')
                rejected[reason]+=1;continue
            # answer remains the existing unused placeholder in this zero-update experiment.
            selected.append({**{k:v for k,v in row.items() if k not in ('student_prompt_ids','teacher_prompt_ids')},
                             'question_hash':sha,'answer':row.get('answer','0')})
            seen_ids.add(row['id']);seen_questions.add(sha)
    if len(selected)!=count:
        raise ValueError(f'Only {len(selected)} distinct prompt-valid math questions; requested {count}')
    return selected,dict(rejected)


def local_candidates(folder, count, seed, *, excluded_ids=()):
    """Uniform row sampling over all local math rows, before question deduplication."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    shards=[];ends=[];math_total=global_offset=0
    for path in sorted(folder.glob('*.parquet')):
        pf=pq.ParquetFile(path)
        domains=pf.read(columns=['domain'])['domain'].to_pylist()
        math_rows=[i for i,v in enumerate(domains) if v=='math']
        if math_rows:
            group_ends=[];end=0
            for i in range(pf.metadata.num_row_groups):
                end+=pf.metadata.row_group(i).num_rows;group_ends.append(end)
            shards.append({'path':path,'math_rows':math_rows,'global_offset':global_offset,'group_ends':group_ends})
            math_total+=len(math_rows);ends.append(math_total)
        global_offset+=pf.metadata.num_rows
    if not math_total:raise ValueError('No math rows in the existing local dataset')
    excluded=set(excluded_ids)
    if excluded:
        rng=random.Random(seed);sampled=[];seen=set()
        available=sum(f'ot3m-{i:06d}' not in excluded for i in range(math_total))
        while len(sampled)<min(available,count*5):
            ordinal=rng.randrange(math_total)
            if ordinal in seen or f'ot3m-{ordinal:06d}' in excluded:continue
            seen.add(ordinal);sampled.append(ordinal)
    else:
        sampled=random.Random(seed).sample(range(math_total),min(math_total,count*5))
    groups=defaultdict(list)
    for order,ordinal in enumerate(sampled):
        shard_index=bisect_right(ends,ordinal);shard=shards[shard_index]
        local_row=shard['math_rows'][ordinal-(ends[shard_index-1] if shard_index else 0)]
        group=bisect_right(shard['group_ends'],local_row)
        groups[(shard_index,group)].append((order,ordinal,local_row))
    decoded={};used=[];done=0
    for shard_index,shard in enumerate(shards):
        wanted=[(group,entries) for (si,group),entries in groups.items() if si==shard_index]
        if not wanted:continue
        path=shard['path'];pf=pq.ParquetFile(path)
        stat=path.stat();sha=file_sha256(path)
        for group,entries in wanted:
            begin=shard['group_ends'][group-1] if group else 0
            table=pf.read_row_group(group,columns=['domain','source','difficulty','conversations'])
            table=table.take(pa.array([local-begin for _,_,local in entries]))
            conversations=table['conversations'].combine_chunks();flat=pc.list_flatten(conversations)
            human=pc.equal(pc.struct_field(flat,'from'),'human')
            parents=pc.filter(pc.list_parent_indices(conversations),human).to_pylist()
            texts=pc.filter(pc.struct_field(flat,'value'),human).to_pylist()
            if parents!=list(range(len(entries))):raise ValueError('Expected one human question per sampled OT3 row')
            domains=table['domain'].to_pylist();sources=table['source'].to_pylist();difficulty=table['difficulty'].to_pylist()
            for i,(order,ordinal,local) in enumerate(entries):
                decoded[order]={'id':f'ot3m-{ordinal:06d}','question':texts[i].strip(),'answer':'0',
                    'domain':domains[i],'source':sources[i],'difficulty':difficulty[i],
                    'source_file':path.name,'source_row_in_file':local,
                    'source_global_row':shard['global_offset']+local,'source_math_ordinal':ordinal}
        after=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Dataset shard changed while sampling')
        used.append({'file':path.name,'sha256':sha,'bytes':stat.st_size})
        done+=1
        if done%10==0:print(f'Local math sampling: {done} shards, {len(decoded)}/{len(sampled)} candidate rows',flush=True)
    return [decoded[i] for i in range(len(sampled))],{
        'sampling':('seeded_uniform_math_rows_excluding_existing_ids' if excluded else
                    'seeded_uniform_math_rows_then_deduplicate_questions_and_check_native_prompt_lengths'),
        'seed':seed,'math_rows':math_total,'sampled_candidate_rows':len(sampled),'shards':used}


def prepare(cfg):
    if cfg.get('layout_version')==1:cfg=configure_layout(cfg)
    source=project_path(cfg['question_source']);data=project_path(cfg['data_dir'])
    base_path=project_path(cfg['base_question_source']);base=read_json(base_path)
    codecs=load_codecs(cfg);count=cfg['question_count']
    manifest_path=data/'sampling_manifest.json'
    input_dir=project_path(cfg['input_data_dir']) if cfg.get('input_data_dir') else None
    imported=None
    if not source.exists() and input_dir is not None and (input_dir/'questions_source.json').exists():
        import shutil
        imported=read_json(input_dir/'questions_source.json')
        if len(imported)!=count:
            raise ValueError('INPUT_DATA_DIR question count differs; select the matching frozen input directory')
        source.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(input_dir/'questions_source.json',source)
        if (input_dir/'sampling_manifest.json').exists():
            shutil.copyfile(input_dir/'sampling_manifest.json',manifest_path)
        else:
            frozen_json(manifest_path,{'schema':'frozen_input_copy_v1','questions':count,
                'base_source_sha256':file_sha256(base_path),'question_source_sha256':file_sha256(source),
                'source':str(input_dir),'preserved_base_questions':len(base),'added_questions':count-len(base)})
    if source.exists():
        saved=read_json(manifest_path)
        if (saved['question_source_sha256']!=file_sha256(source) or saved['base_source_sha256']!=file_sha256(base_path)
                or saved['questions']!=count):raise ValueError('Frozen expanded data identity changed')
        questions=read_json(source)
    else:
        # Shared assets are read-only links; only generated artifacts use project_path.
        if cfg.get('data_policy')=='all_train':
            candidates,provenance=local_candidates(ROOT/cfg['local_dataset'],max(1,count-len(base)),cfg['seed'],
                excluded_ids={q['id'] for q in base})
            from steer_opd.training.benchmarks import benchmark_rows
            benchmarks,_=benchmark_rows(base)
            reserved={' '.join(r['extra_info']['question'].split()) for r in benchmarks}
            candidates=[q for q in candidates if ' '.join(q['question'].split()) not in reserved]
        else:
            candidates,provenance=local_candidates(ROOT/cfg['local_dataset'],count,cfg['seed'])
        questions,rejected=select_expansion(base,candidates,count,codecs,cfg['max_prompt_tokens'],cfg['enable_thinking'])
        frozen_json(source,questions)
        frozen_json(manifest_path,{'schema':'local_ot3_expansion_v1','created_at':timestamp(),
            'source_dataset':'open-thoughts/OpenThoughts3-1.2M','dataset_revision':cfg['dataset_revision'],
            'questions':count,'preserved_base_questions':len(base),'added_questions':count-len(base),
            'base_source':cfg['base_question_source'],'base_source_sha256':file_sha256(base_path),
            'question_source_sha256':file_sha256(source),'preparation_code_sha256':file_sha256(Path(__file__)),
            'max_prompt_tokens':cfg['max_prompt_tokens'],'rejected_candidates':rejected,
            'answer_policy':'Existing unused placeholder; no accuracy grading or labels inferred',**provenance})
    expected={r['id']:r['question'] for r in base};actual={r['id']:r['question'] for r in questions}
    if len(questions)!=count or any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError('Expanded pool must preserve the original questions')
    prepared=prepare_questions(cfg,codecs)
    if cfg.get('data_policy')=='all_train':
        print({'questions':len(prepared),'preserved_base':len(base),'training_questions':len(prepared),
            'data_policy':'all_train','source':str(source),'gpu_started':False},flush=True)
        return
    split=read_json(data/'split.json')
    if imported is not None and (input_dir/'split.json').exists():
        old=read_json(input_dir/'split.json')
        if cfg['select_fraction']==.7 and any(split[k]!=old[k] for k in ('select_ids','confirm_ids')):
            raise ValueError('The original frozen 70/30 split must be preserved')
    print({'questions':len(prepared),'preserved_base':len(base),'select':len(split['select_ids']),
           'confirm':len(split['confirm_ids']),'source':str(source),'gpu_started':False},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ablation.json')
    prepare(read_json(project_path(parser.parse_args().config)))
