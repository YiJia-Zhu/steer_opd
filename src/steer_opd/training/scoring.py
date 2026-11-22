"""Exact-text, checkpoint-bound scoring requests for a separate EasySteer process."""
from pathlib import Path
import torch
from transformers import AutoTokenizer

from steer_opd.checkpoint_identity import checkpoint_content_id
from steer_opd.compass_alignment import ByteLevelCodec
from steer_opd.direction_bank import file_sha256
from steer_opd.math_data import prompt_ids
from steer_opd.resumable_core import ROOT, read_json
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.losses import training_alignment
from steer_opd.training.rollout_logprobs import STOPPED_LENGTH_FIELD


def codecs_for_plan(plan):
    paths = {**plan['tokenizer_paths']}
    if plan['training_loss']['loss_type'] == 'compass_opd':
        paths['reference'] = plan['training_loss']['reference_model']
    return {role: ByteLevelCodec(AutoTokenizer.from_pretrained(ROOT/path, local_files_only=True))
            for role, path in paths.items()}


def validate_assets(plan):
    s = plan['settings']
    path = ROOT/s['sv_bank']
    bank = torch.load(path, map_location='cpu', weights_only=True)
    model_ids = {role: checkpoint_content_id(ROOT/plan[role]) for role in ('student', 'teacher')}
    if (bank['identity']['checkpoint_content_id'] != model_ids['teacher']
            or bank['identity']['step'] != 0 or bank['method']['direction'] != 'S-T'
            or bank['method']['granularity'] != 'global' or list(bank['method']['layers']) != [s['teacher_layer']]):
        raise ValueError('Teacher global bank layer, checkpoint or direction does not match training')
    if plan.get('data_policy')=='all_train':
        identity=bank['identity']
        if (identity.get('data_policy')!='all_train'
                or identity.get('question_source_sha256')!=plan['question_source_sha256']
                or identity.get('training_ids_sha256')!=plan['training_ids_sha256']):
            raise ValueError('Teacher SV bank must come from the same complete training question pool')
    if plan['training_loss']['loss_type'] == 'compass_opd':
        reference = ROOT/plan['training_loss']['reference_model']
        model_ids['reference'] = checkpoint_content_id(reference)
        def family(path):
            cfg=read_json(path/'config.json');cfg=cfg.get('text_config',cfg)
            return cfg['model_type'].replace('_moe','')
        if family(reference) != family(ROOT/plan['teacher']):
            raise ValueError('Ref must belong to the teacher release family')
    from steer_opd.stage_contracts import codec_key
    return {'model_content_ids': model_ids, 'sv_bank_sha256': file_sha256(path),
            'codecs':{role:codec_key(codec) for role,codec in codecs_for_plan(plan).items()}}


def batch_rows(batch):
    response=batch.batch['responses'];width=response.shape[1]
    attention=batch.batch['attention_mask']
    generated=batch.batch['response_mask']
    stopped_lengths=batch.non_tensor_batch.get(STOPPED_LENGTH_FIELD)
    rows=[]
    for i in range(len(response)):
        info=batch.non_tensor_batch['extra_info'][i]
        n=int(attention[i,-width:].sum())
        prompt=batch.batch['prompts'][i][attention[i,:-width].bool()].tolist()
        row={'id':str(batch.non_tensor_batch['uid'][i]),
            'question_id':str(info['question_id']), 'question':info['question'],
            'prompt_token_ids':prompt, 'token_ids':response[i,:n].tolist()}
        if n and stopped_lengths is not None and stopped_lengths[i] == n and generated[i,n-1]:
            row['sampled_stop_id'] = row['token_ids'][-1]
        rows.append(row)
    return rows


def prepare_request(plan, rows, codecs):
    settings=TrainingLossConfig(**plan['training_loss'])
    prepared=[]
    for row in rows:
        expected=prompt_ids(codecs['student'].tokenizer,row['question'],enable_thinking=False)
        if expected!=row['prompt_token_ids']:
            raise ValueError('VERL student prompt differs from the frozen math/chat template')
        alignment=training_alignment(row['token_ids'],codecs['student'],codecs['teacher'],settings=settings,
                                     reference_codec=codecs.get('reference'),
                                     sampled_stop_id=row.get('sampled_stop_id'))
        native={role:{'id':row['id'],'key':row['id'],
                'prompt_token_ids':prompt_ids(codec.tokenizer,row['question'],enable_thinking=False),
                'token_ids':alignment['native_response_ids'][role]}
                for role,codec in codecs.items() if role!='student'}
        if settings.loss_type=='compass_opd':
            native['student_anchor']={'id':row['id'],'key':row['id'],
                'prompt_token_ids':row['prompt_token_ids'],'token_ids':row['token_ids']}
        prepared.append({'source':row,'alignment':alignment,'native':native})
    # The array manifest binds all native inputs and full alignment contents.
    # Avoid serializing the large per-span dictionary a second time for a hash.
    return {'plan_binding':plan['binding'],'rows':prepared,'seed':42}
