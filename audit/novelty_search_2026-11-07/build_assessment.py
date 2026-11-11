from pathlib import Path
import json,csv,datetime,re
ROOT=Path(__file__).resolve().parent
project=ROOT.parent.parent
pool=json.loads((ROOT/'search_pool.json').read_text())
metadata={r['version']:r for r in json.loads((ROOT/'selected_metadata.json').read_text())['records']}
sources=json.loads((ROOT/'source_manifest.json').read_text())['sources']
# Labels refer to evidence in the requested fixed version, not all possible uses.
# P: direct_cross_family, cross_tokenizer_interface, adjacent_same_lineage,
#    same_model_or_no_cross_pair, not_checked.
# B: activation_opd, representation_auxiliary, output_guidance,
#    latent_context_opd, activation_without_opd, no_steering, not_checked.
labels={
 '2606.09456v1':('cross_tokenizer_interface','no_steering','overview','§3–§4方法概述；chunk对齐与信用分配'),
 '2607.22334v1':('cross_tokenizer_interface','no_steering','overview','§3字节前缀映射、残余质量及适用条件'),
 '2605.07711v3':('cross_tokenizer_interface','no_steering','overview','§3共同监督空间与最小对齐单位'),
 '2610.13257v1':('cross_tokenizer_interface','no_steering','overview','chunk内信用分配与未对齐位置锚定；模型配对摘要'),
 '2610.08448v1':('cross_tokenizer_interface','no_steering','overview','strict/top-k与span监督可靠性的正文概述'),
 '2609.34738v1':('cross_tokenizer_interface','no_steering','overview','字节事件进入与completion-set监督的正文概述'),
 '2609.33816v1':('cross_tokenizer_interface','no_steering','overview','§3新输出head、PTS/HPA；冻结教师主干'),
 '2610.27555v1':('direct_cross_family','no_steering','overview','原生安全跨模型转移与cross-tokenizer OPD；不等同SafeSteer'),
 '2606.06021v4':('direct_cross_family','representation_auxiliary','method_and_setup','§3.3固定低秩桥；§4.2.4 Phi→Qwen异构tokenizer结果'),
 '2610.23306v1':('adjacent_same_lineage','representation_auxiliary','method_and_setup','§3双memory bank相似度KL；§4.1配对均为Qwen3'),
 '2610.02381v1':('adjacent_same_lineage','representation_auxiliary','method_and_setup','§3式(2)(3)短程表示辅助；§4.1定义cross-family仍为Qwen backbone'),
 '2610.26397v2':('cross_tokenizer_interface','no_steering','overview','外部文本教师与学生家族自教师桥接；非固定激活方向'),
 '2606.02530v2':('same_model_or_no_cross_pair','activation_opd','method_and_setup','§3.1加拒绝方向到教师；§3.4学生rollout上局部词表反向KL'),
 '2605.12400v2':('same_model_or_no_cross_pair','output_guidance','method_and_setup','§4.4成功/失败条件logits差，重构目标后OPSD'),
 '2610.14001v1':('same_model_or_no_cross_pair','representation_auxiliary','method_and_setup','§2.3低秩shift投影与题目路由；PDF正文'),
 '2609.12579v1':('same_model_or_no_cross_pair','representation_auxiliary','method_and_setup','§3.1–§3.2固定rank64辅助与Fisher；不做前向加向量'),
 '2609.36484v2':('same_model_or_no_cross_pair','representation_auxiliary','method_and_setup','§4.2动态RL残差目标；§5.1每个pair共享架构/tokenizer/head'),
 '2605.22675v2':('same_model_or_no_cross_pair','activation_without_opd','method_and_setup','§3.3投影KV生成语料，再无hook做普通SFT'),
 '2606.29340v1':('same_model_or_no_cross_pair','representation_auxiliary','overview','token转移方向与几何辅助；未逐一复核推导'),
 '2608.13040v1':('same_model_or_no_cross_pair','latent_context_opd','method_and_setup','§3.2 latent composer；§3.3冻结同底座教师与OPSD'),
 '2611.08016v1':('not_checked','no_steering','overview','摘要/引言及§3.2–§3.4机制定位；未完整复核response-law推导'),
 '2611.07241v1':('not_checked','output_guidance','abstract','摘要：输出目标的方向/距离解耦；未将其视为activation steering'),
 '2611.07731v1':('not_checked','no_steering','abstract','摘要：前缀监督、support/tail与overlap gate；未逐公式复核'),
 '2605.13230v2':('not_checked','output_guidance','abstract','摘要：教师指导token生成＋结果奖励；不是已核实的hidden steering'),
 '2607.13399v1':('not_checked','no_steering','abstract','摘要：mismatch和长度病理，advantage clipping/log compression'),
 '2609.33455v2':('not_checked','no_steering','overview','普通rollout后student训练上下文dropout、teacher保持完整；不是方向注入'),
 '2606.07082v3':('not_checked','no_steering','abstract','几何分析候选；本轮不以标题推定activation steering'),
 '2609.28845v1':('adjacent_same_lineage','representation_auxiliary','method_and_setup','§4失效、§5最后层表示辅助前10步crossfade、§6.1模型'),
 '2608.28859v1':('same_model_or_no_cross_pair','activation_without_opd','overview','§5将steered activation内化；不是外部跨family输出OPD'),
 '2610.16288v2':('not_checked','activation_without_opd','method_and_setup','§2.3–§2.4目标模型replay教师文本；目标冻结且推理保留steering'),
 '2608.09836v1':('adjacent_same_lineage','output_guidance','abstract','摘要：Hellinger excess与teacher top-K deficit修正'),
 '2607.06855v2':('same_model_or_no_cross_pair','no_steering','abstract','摘要：输出Fisher-Rao/Hellinger几何；多family独立自蒸馏不等于跨family配对'),
 '2605.22263v1':('same_model_or_no_cross_pair','output_guidance','abstract','摘要：按熵选择吸引/排斥教师的输出方向'),
 '2609.37044v2':('adjacent_same_lineage','no_steering','method_and_setup','§3兼容性TRD与收益routing；§4.1实际跨模型配对仍是Qwen3'),
 '2606.11709v2':('not_checked','output_guidance','overview','正确/错误hint对比消除style drift；外部跨family附录未深入核查'),
}
assert set(labels)==set(metadata)=={s['version'] for s in sources}
assessed=[]
for source in sources:
 v=source['version'];p,b,depth,scope=labels[v];r=metadata[v]
 assessed.append({'version':v,'title':r['title'],'published':r['published'],'updated':r['updated'],
  'abstract_url':'https://arxiv.org/abs/'+v,'source_text':source['text_file'],
  'problem_axis':p,'mechanism_axis':b,'reading_depth':depth,'reading_scope':scope,
  'exact_current_pipeline_match':'not_identified_in_reviewed_scope'})
queries=[]
for p in sorted(ROOT.glob('*.json')):
 d=json.loads(p.read_text())
 if isinstance(d,dict) and 'query' in d and 'records' in d and 'total_results' in d:
  queries.append({'file':p.name,'query':d['query'],'total_results':d['total_results'],'returned':len(d['records']),'truncated':d['total_results']>len(d['records']),'status':d.get('status')})
assert len(queries)==16 and len(pool)==530
core=[]
for path in [project/'papers/manifest.json',project/'ref/manifest.json']:
 for p in json.loads(path.read_text())['papers']:core.append({'version':p['version'],'title':p['title'],'source_text':p['source_text'],'manifest':str(path.relative_to(project))})
assessment={'assembled_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cutoff_date_utc':'2026-11-07',
 'task':'Novelty audit only; cross-family OPD and steering+OPD; no model/experiment changes.',
 'arxiv_queries':queries,'unique_metadata_candidates':len(pool),'additional_fulltext_sources':len(sources),
 'evidence_depth_counts':{k:sum(r['reading_depth']==k for r in assessed) for k in ['method_and_setup','overview','abstract']},
 'scope_limits':['Keyword retrieval is not an exhaustive global novelty proof.','Fulltext retrieval count is not a count of papers read end-to-end.','Broad directions and title-steer queries are truncated; narrower complete queries supplement them.','Bing RSS returned unrelated results and is excluded from substantive evidence.'],
 'conclusion':'Both broad areas have prior art. No complete match to the specified fixed-student-direction, training-forward-only, early-withdrawal pipeline was identified in the reviewed scope. Strong component and conceptual overlap remains.',
 'papers':assessed,'reused_local_primary_sources':core}
(ROOT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n')
selected={re.sub(r'v\d+$','',r['version']):r for r in assessed};existing={re.sub(r'v\d+$','',r['version']):r for r in core}
fields=['arxiv_id','version','title','matched_queries','triage','reading_depth','problem_axis','mechanism_axis','notes']
with (ROOT/'screening.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for id,r in sorted(pool.items()):
  s=selected.get(id);e=existing.get(id)
  if s:triage='additional_source_selected';depth=s['reading_depth'];problem=s['problem_axis'];mechanism=s['mechanism_axis'];note=s['reading_scope']
  elif e:triage='existing_local_primary';depth='see_existing_notes';problem=mechanism='see_existing_notes';note=e['source_text']
  else:
   triage='metadata_keyword_triage_only';depth='not_method_checked';problem=mechanism='not_assessed';note='Search hit only; no method-level inclusion or exclusion claim.'
  w.writerow({'arxiv_id':id,'version':r['version'],'title':r['title'],'matched_queries':'|'.join(r['matched_queries']),
   'triage':triage,'reading_depth':depth,'problem_axis':problem,'mechanism_axis':mechanism,'notes':note})
print('Queries',len(queries),'candidates',len(pool),'sources',len(sources),'reading depths',assessment['evidence_depth_counts'])
