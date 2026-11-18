"""Scientific dependency keys; injection parameters never invalidate source rollouts."""
import json
import subprocess

from steer_opd.artifact_reuse import contract
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import digest
from steer_opd.method_config import DIRECTION_VERSIONS, reader_direction, reader_layers

GEN_CODE=('scripts/generate_static_sources.py','src/steer_opd/rollout_topk.py',
          'src/steer_opd/easysteer_static.py','src/steer_opd/easysteer_worker.py',
          'src/steer_opd/generation_policy.py')
CAP_CODE=('src/steer_opd/extraction.py','src/steer_opd/easysteer_extraction.py','src/steer_opd/easysteer_static.py',
          'src/steer_opd/easysteer_worker.py')
BANK_CODE=('src/steer_opd/compass_banks.py','src/steer_opd/direction_bank.py')
ALIGN_CODE=('src/steer_opd/compass_alignment.py',)
NATIVE_CODE=('scripts/score_compass_sources.py','src/steer_opd/easysteer_static.py',
             'src/steer_opd/easysteer_worker.py')
METRIC_CODE=('src/steer_opd/compass_alignment.py','src/steer_opd/prefix_matching.py')


def codec_key(codec):
    return {'mapping':codec.mapping_sha256,'support':codec.support_id,'eos':sorted(codec.eos),
            'special':sorted(codec.special),'template':digest(codec.tokenizer.chat_template)}


def runtime_key(python):
    """Query package metadata in the engine environment, without importing CUDA/vLLM."""
    program="import importlib.metadata,json;names=['torch','transformers','vllm','easysteer','tokenizers'];print(json.dumps({n:importlib.metadata.version(n) for n in names}))"
    return json.loads(subprocess.check_output([str(python),'-c',program],text=True))


def generation(cfg,role,questions,codec,model_id,runtime):
    from steer_opd.generation_policy import presence_penalty
    source=role=='student'
    requests=[{'id':q['id'],'prompt':prompt_ids(codec.tokenizer,q['question'],cfg['enable_thinking']),
               'seed':42} for q in questions]
    return contract(role,{'requests':digest(requests),'model_content_id':model_id,'codec':codec_key(codec),
        'decode_cuda_graph':cfg.get('decode_cuda_graph',True),'head_dtype':'float32',
        'max_tokens':cfg['source_max_tokens'] if source else cfg['reference_max_tokens'],
        'temperature':1. if source else cfg['reference_temperature'],
        'top_p':1. if source else cfg['reference_top_p'],'top_k':-1,'presence_penalty':presence_penalty(cfg),
        'seed':42,'dtype':'bfloat16','logprobs':16 if source else None,'normalization':'raw',
        'tensor_parallel_size':1 if source else cfg['teacher_gpus'],
        'gpu_memory_utilization':cfg['gpu_memory_utilization'],'batch_invariant':source,
        'runtime':runtime},code=GEN_CODE)


def capture(cfg,role,model_id,codec,parents,runtime):
    direction=reader_direction(role)
    code=CAP_CODE+(('scripts/capture_student_activations.py',) if cfg.get('data_policy')=='all_train' else ())
    return contract('capture_'+role,{'model_content_id':model_id,'codec':codec_key(codec),
        'layers':reader_layers(cfg,role),'context_len':cfg['sv_max_context_len'],
        'pooling':'sv_training_all_retained_answer_tokens','direction_version':DIRECTION_VERSIONS[direction],
        'tensor_parallel_size':cfg['teacher_gpus'] if role=='teacher' else 1,
        'gpu_memory_utilization':cfg['gpu_memory_utilization'],'backend':cfg['extraction_backend'],
        'batch_invariant':role=='student','runtime':runtime},parents=parents,code=code)


def bank(cfg,role,own_capture,common_ids):
    code=BANK_CODE+(('scripts/prepare_compass_banks.py',) if cfg.get('data_policy')=='all_train' else ())
    return contract('bank_'+role,{'direction':reader_direction(role),'layers':reader_layers(cfg,role),
        'valid_question_ids':list(common_ids),'pooling_and_scaling':'raw_difference_mean_before_unit_norm_own_reader_median_norm',
        'injection_alpha_is_not_a_bank_dependency':True},parents={'capture':own_capture},code=code)


def alignment(cfg,source,codecs):
    return contract('alignment',{'codecs':{r:codec_key(c) for r,c in codecs.items()}},
                    parents={'student_rollout':source},code=ALIGN_CODE)


def native(cfg,model_id,codec,parents,runtime):
    return contract('native_teacher',{'model_content_id':model_id,'codec':codec_key(codec),'top_k':16,
        'raw_probabilities':True,'backend':cfg['probability_backend'],'tensor_parallel_size':cfg['teacher_gpus'],
        'gpu_memory_utilization':cfg['gpu_memory_utilization'],'batch_invariant':False,'runtime':runtime},
        parents=parents,code=NATIVE_CODE)


def metrics(parents):
    from steer_opd.prefix_matching import METRIC_VERSION,RULE_VERSION,P_DEFINITION
    return contract('source_metrics',{'metric_version':METRIC_VERSION,'matching_rule':RULE_VERSION,
                                      'P_overlap_definition':P_DEFINITION},parents=parents,code=METRIC_CODE)


def calibration(cfg,parents,select_ids):
    return contract('calibration',{'gate_metric':cfg['gate_metric'],'target_rate':cfg['gate_target_rate'],
        'select_ids':sorted(select_ids),'strict_lt':True,'both_metrics_calibrated':True},
        parents=parents,code=('src/steer_opd/overlap_tuning.py',))


def screen(cfg,role,conditions,parents,rows):
    return contract('screen_'+role,{'conditions':conditions,'row_prefixes':digest([
        {k:r[k] for k in ('id','key','prompt_token_ids','token_ids')} for r in rows]),
        'rank_metric':cfg['rank_metric'],'backend':cfg['screen_backend'],
        'runtime':cfg.get('_screen_runtime'), 'teacher_gpus':cfg['teacher_gpus'],
        'gpu_memory_utilization':cfg['gpu_memory_utilization'],
        'batch_size':cfg['student_score_batch_size'] if role=='student' else cfg['teacher_score_batch_size']},parents=parents,
        code=('scripts/run_overlap_ablation.py','src/steer_opd/teacher_screen.py',
              'src/steer_opd/static_screen.py','src/steer_opd/operators.py','src/steer_opd/easysteer_plugin.py',
              'src/steer_opd/cpu_metrics.py','src/steer_opd/raw_additive.py'))
