"""Independent exploration supervisor. Only GPU0/1; serial stages, complete-call resume."""
from common import *
import argparse,fcntl,os,signal,subprocess,sys,shutil,time,csv
from steer_opd.experiment_runtime import timestamp


def identity():
 return {str(p.relative_to(ROOT/'exploration')):file_sha256(p) for p in sorted((ROOT/'exploration').rglob('*.py')) if '__pycache__' not in p.parts}


def run_stage(phase,cases,config):
 if not cases:return
 if shutil.disk_usage(ROOT).free < config['disk_reserve_for_parameter_search_bytes']+5*2**30:
  raise OSError('Exploration paused at storage guard to preserve parameter-search headroom')
 path=OUT/phase;path.mkdir(parents=True,exist_ok=True)
 command=[str(ROOT/'.venv-easysteer/bin/python'),str(ROOT/'exploration/runner.py'),'--phase',phase,'--cases',','.join(cases)]
 env=dict(os.environ);env.update(PYTHONPATH=str(SNAPSHOT)+os.pathsep+str(ROOT/'.runtime/easysteer-plugin'),STEER_OPD_METRIC_WORKERS='8')
 launch=[sys.executable,str(ROOT/'exploration/run_on_idle_gpus.py'),'--count','2','--gpu-ids','0,1','--wait-seconds','3600','--run-dir',str(path/'admission'/str(time.time_ns())),'--',*command]
 with (path/'stage.console.log').open('ab',buffering=0) as output:
  child=subprocess.Popen(launch,cwd=ROOT/'exploration',env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait()
  except BaseException:
   child.terminate()
   try:child.wait(timeout=45)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   raise
 if code:raise subprocess.CalledProcessError(code,launch)


def aggregate(config):
 rows=[]
 for phase in ('pilot','select','confirm'):
  path=OUT/phase/'summaries.json'
  if not path.exists():continue
  for case,v in read_json(path).items():
   for metric,m in v['metrics'].items():
    rows.append({'phase':phase,'pilot_only':phase=='pilot','case':case,'metric':metric,'questions':m['questions'],
     'value':m['question_weighted_value'],'delta_pp':m['question_weighted_delta_pp'],
     'token_value':m.get('token_weighted_value',m.get('token_weighted_overlap16')),'token_delta_pp':m['token_weighted_delta_pp']})
 if rows:
  SUMMARY.mkdir(parents=True,exist_ok=True)
  with (SUMMARY/'comparison.csv').open('w',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return rows


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--status',action='store_true');args=parser.parse_args()
 config=read_json(ROOT/'exploration/config.json');SUMMARY.mkdir(parents=True,exist_ok=True)
 if args.status:print(read_json(SUMMARY/'progress.json'));return
 print({'experiment':'exploration','gpu_ids':[0,1],'pilot':64,'full':[700,300],'candidates':list(config['candidates'])},flush=True)
 if not args.run:return
 lock=(SUMMARY/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signature={'config_sha256':file_sha256(ROOT/'exploration/config.json'),'source_hashes':identity()}
 manifest=SUMMARY/'execution_identity.json'
 if manifest.exists() and read_json(manifest)!=signature:raise ValueError('Frozen exploration identity changed')
 if not manifest.exists():atomic_json(manifest,signature)
 def stopped(*args):raise KeyboardInterrupt('exploration interrupted')
 signal.signal(signal.SIGTERM,stopped)
 state={'status':'running','pid':os.getpid(),'started_at':timestamp(),'gpu_ids':[0,1],'phase':'pilot','parameter_updates':0}
 atomic_json(SUMMARY/'progress.json',state)
 try:
  run_stage('pilot',['no_sv_replay',*config['candidates']],config);aggregate(config)
  if identity()!=signature['source_hashes']:raise ValueError('Exploration source changed')
  pilot=read_json(OUT/'pilot/summaries.json');ref=pilot['reference_replay']['metrics']['overlap']['question_weighted_delta_pp']
  eligible=[name for name,c in config['candidates'].items() if c['eligible']]
  ranked=sorted(eligible,key=lambda name:(-pilot[name]['metrics']['overlap']['question_weighted_delta_pp'],name))
  selected=[name for name in ranked if pilot[name]['metrics']['overlap']['question_weighted_delta_pp']>=ref+.05
   and pilot[name]['metrics']['P_overlap']['question_weighted_delta_pp']>=0][:2]
  if not selected and pilot[ranked[0]]['metrics']['overlap']['question_weighted_delta_pp']>0:selected=ranked[:1]
  atomic_json(SUMMARY/'selected_candidates.json',{'selected':selected,'ranked':ranked,'uses_confirm':False,
    'selection_scope':'pilot64 from original Select only','pilot_reference_delta_pp':ref})
  for phase in ('select','confirm'):
   if identity()!=signature['source_hashes']:raise ValueError('Exploration source changed')
   state.update(phase=phase,selected=selected,updated_at=timestamp());atomic_json(SUMMARY/'progress.json',state)
   run_stage(phase,selected,config);aggregate(config)
  state.update(status='complete',finished_at=timestamp());atomic_json(SUMMARY/'progress.json',state)
 except BaseException as e:
  aggregate(config);state.update(status='failed_resumable',error=repr(e),finished_at=timestamp());atomic_json(SUMMARY/'progress.json',state);raise
if __name__=='__main__':main()
