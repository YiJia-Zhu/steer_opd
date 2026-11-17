from common import *
import random,torch
from directions import build_directions
from steer_opd.artifact_reuse import StageArtifacts

def main():
 torch.set_num_threads(8);torch.manual_seed(42)
 # Validate committed banks/captures and result descriptors. Nothing in the owner is changed.
 cache=StageArtifacts({'output_root':'outputs'});provenance={}
 for name,p in [('pool',POOL),('bank29',BANK29),('bank19',BANK19)]:
  cache.validate(p.parent/'metadata/artifact.json')
  provenance[name]={'path':str(p.relative_to(ROOT)),'sha256':file_sha256(p)}
 split=read_json(DATA/'split.json');pools=load_packet(POOL)['pooled'];ids=list(pools)
 if len(ids)!=1000 or set(ids)!=set(split['select_ids']+split['confirm_ids']):raise ValueError('Changed 1000-question SV pool')
 if not all(pools[q]['usable'] for q in ids):raise ValueError('Unexpected invalid pool')
 a=torch.stack([pools[q]['layers']['29']['pools'][0] for q in ids]);b=torch.stack([pools[q]['layers']['29']['pools'][1] for q in ids])
 original=load_packet(BANK29);torch.testing.assert_close((a-b).mean(0),original['layers']['29']['direction'],rtol=0,atol=0)
 specs,vectors,geometry=build_directions(ids,a,b,load_packet(BANK19)['layers']['19']['direction'])
 atomic_torch(VECTORS/'directions.pt',vectors)
 shuffled=sorted(split['select_ids']);random.Random(42).shuffle(shuffled);pilot=shuffled[:64]
 config={'experiment':'exploration','name':NAME,'seed':42,'question_count':1000,'pilot_select_ids':pilot,
  'original_split':{'select_ids':split['select_ids'],'confirm_ids':split['confirm_ids']},
  'teacher':'assets/models/Qwen3.5-35B-A3B','student':'assets/models/Qwen3-4B',
  'metric_version':4,'returned_top_k':16,'P_overlap_definition':'top16_raw_probability_maximum_prefix_compatible_mass',
  'candidates':specs,'directions_path':str((VECTORS/'directions.pt').relative_to(ROOT)),
  'directions_sha256':file_sha256(VECTORS/'directions.pt'),'sv_geometry':geometry,'inputs':provenance,
  'data_sha256':{name:file_sha256(DATA/name) for name in ('questions_source.json','questions.json','split.json')},
  'reuse_pilot_from':str((OUT.with_name('st_direction_design_n1000_seed42_v2')).relative_to(ROOT)),
  'reuse_pilot_config':str((SUMMARY.with_name('st_direction_design_n1000_seed42_v2')/'frozen_config.json').relative_to(ROOT)),
  'parameter_updates':0,'gpu_ids':[0,1],'screening_selection':'pilot Select question-average delta_overlap only; no Confirm ranking',
  'full_evaluation_policy':'top 2 eligible candidates beating reference pilot by >=0.05pp; otherwise top1 if positive',
  'full_stage_counts':{'select':700,'confirm':300},'target_delta_pp':2.,'max_full_candidates':2,
  'disk_reserve_for_parameter_search_bytes':60*2**30,'tensor_compression':'gzip'}
 for split_name in ('select','confirm'):
  cache.validate(ST/split_name/'teacher/metadata/artifact.json')
  job=read_json(ST/f'metadata/jobs/{split_name}_teacher.json')
  provenance[f'{split_name}_job']={'path':str((ST/f'metadata/jobs/{split_name}_teacher.json').relative_to(ROOT)),
    'sha256':file_sha256(ST/f'metadata/jobs/{split_name}_teacher.json')}
  if len(read_json(input_path(job['rows'])))!=(700 if split_name=='select' else 300):raise ValueError('Wrong frozen split')
 atomic_json(ROOT/'exploration/config.json',config);atomic_json(SUMMARY/'geometry.json',geometry)
 print('Prepared independent exploration:',len(specs),'pilot Select64; full uses original700/300',flush=True)
 print(geometry,flush=True)
if __name__=='__main__':main()
