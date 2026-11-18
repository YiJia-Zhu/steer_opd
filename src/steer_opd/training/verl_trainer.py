"""Synchronous shared-two-GPU VERL extension; optimizer/checkpointing stay upstream."""
from steer_opd.resumable_core import ROOT,atomic_json
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.verl_adapter import attach_advantages,verl_loss,register_token_reward_direct
from steer_opd.training.scoring import batch_rows
from steer_opd.training.scorer_client import ScorerClient
from steer_opd.training.shared_pool import SharedCheckpointManager,SharedPoolHandover
from steer_opd.training.telemetry import phase,event
from steer_opd.training.rollout_logprobs import SOURCE as STUDENT_LOGPROB_SOURCE,bind_old_log_probs
from steer_opd.generation_policy import validate_training_sampling_policy
from steer_opd.retention import minimal


def serializable_metrics(value):
    """Preserve numeric values when upstream returns NumPy scalar metrics."""
    import numpy as np
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:serializable_metrics(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [serializable_metrics(item) for item in value]
    return value


def task_runner_class():
    import ray
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class SteerTrainer(RayPPOTrainer):
        def init_workers(self):
            self.scorer.ensure_asleep()
            self.log_folder=ROOT/self.steer_plan['output']/'logs'
            self.overlap_monitor=None
            if self.steer_plan['settings'].get('monitor_overlap'):
                from steer_opd.training.overlap_monitor import OverlapMonitor
                self.overlap_monitor=OverlapMonitor(self.steer_plan,self.scorer.codecs)
            if 'tensorboard' in self.config.trainer.logger and not minimal(self.steer_plan):
                from torch.utils.tensorboard import SummaryWriter
                self.validation_generations_logger.writer=SummaryWriter(str(self.log_folder/'tensorboard/samples'),flush_secs=10)
            with phase(self.log_folder,'student_model_and_rollout_initialization'):super().init_workers()
            from steer_opd.training.rollout_seeds import SeededRolloutManager
            self.async_rollout_manager=SeededRolloutManager(self.async_rollout_manager,self.steer_plan)
            # Upstream ends init_workers with all student rollout replicas asleep.
            self.checkpoint_manager=SharedCheckpointManager(self.checkpoint_manager,self.scorer,self.log_folder)
            self.handover=SharedPoolHandover(self.actor_rollout_wg,self.checkpoint_manager,self.scorer,
                ROOT/self.steer_plan['output']/'logs/shared_gpu_handover.jsonl')
            with phase(self.log_folder,'student_initial_offload'):
                self.actor_rollout_wg.to('cpu',model=True,optimizer=True,grad=True)

        def _compute_old_log_prob(self,batch):
            raise RuntimeError('Extra FSDP old-policy forward is disabled; reuse vLLM rollout_log_probs')

        def _update_actor(self,batch):
            with phase(self.log_folder,'student_rollout_logprob_reuse',step=self.global_steps,
                       source=STUDENT_LOGPROB_SOURCE):
                bind_old_log_probs(batch)
            rows=batch_rows(batch)
            expected=self.steer_plan['settings']['train_batch_size']*self.steer_plan['settings']['rollout_n']
            if len(rows)!=expected:
                raise RuntimeError(f'Training rollout batch has {len(rows)} trajectories; expected {expected}')
            with phase(self.log_folder,'teacher_scoring_phase',step=self.global_steps,trajectories=len(rows)):
                with self.handover.frozen_scoring():scored=self.scorer.score(rows)
            overlap_metrics={}
            if self.overlap_monitor is not None:
                with phase(self.log_folder,'cpu_top16_overlap',step=self.global_steps):
                    overlap_metrics=self.overlap_monitor.compute(batch,rows,scored,self.global_steps)
            attach_advantages(batch,scored,TrainingLossConfig(**self.steer_plan['training_loss']))
            self.handover.require_student()
            with phase(self.log_folder,'student_update',step=self.global_steps,
                       valid_tokens=batch.meta_info['steer_opd_valid_tokens']):
                result=super()._update_actor(batch)
            for key,value in overlap_metrics.items():result.meta_info['metrics'][key]=[value]
            import math
            norms=result.meta_info['metrics']['actor/grad_norm']
            if not all(math.isfinite(float(v)) for v in norms):
                raise RuntimeError('Nonfinite gradient norm; refuse to record a successful update')
            event(self.log_folder,'optimizer_update_committed',step=self.global_steps,
                  metrics=result.meta_info.get('metrics',{}))
            if self.steer_plan.get('run_purpose')=='formal_training' and self.global_steps==self.steer_plan.get('resume_source',{}).get('step',0)+1:
                atomic_json(ROOT/self.steer_plan['output']/'metadata/formal_runtime_acceptance.json',{
                    'status':'passed','plan_binding':self.steer_plan['binding'],'actual_step':self.global_steps,
                    'trajectories':len(rows),'loss_type':'traditional_opd','resource_mode':'shared_two_gpu',
                    'student_logprob_source':STUDENT_LOGPROB_SOURCE,
                    'validation_scope':'full-batch generation, scoring, handover and finite optimizer update',
                    'numeric_equivalence':'not_claimed; rollout/actor discrepancy logged per step'})
            self.scorer.release_batch(self.global_steps)
            return result

        def _validate(self,merged=False):
            self.handover.require_student()
            with phase(self.log_folder,'bare_student_benchmark_evaluation',step=self.global_steps):
                metrics=super()._validate(merged=merged)
            if not merged:
                for name in ('MATH-500','AIME2025'):
                    key=f'val-core/steer_eval_{name}/acc/mean@1'
                    if key in metrics:metrics[f'eval/{name}/acc']=metrics[key]
                metrics=serializable_metrics(metrics)
                atomic_json(ROOT/self.steer_plan['output']/f'evaluation/metrics/step_{self.global_steps}.json',metrics)
            return metrics

        def _save_checkpoint(self):
            self.handover.require_student()
            from steer_opd.training.storage import require_training_space
            require_training_space(ROOT/self.steer_plan['output'],48*1024**3)
            checkpoint_root=ROOT/self.steer_plan['output']/'checkpoints'
            try:
                with phase(self.log_folder,'checkpoint_save',step=self.global_steps):
                    result=super()._save_checkpoint()
                    if minimal(self.steer_plan):
                        from steer_opd.training.retention import commit_checkpoint
                        receipt=commit_checkpoint(checkpoint_root,self.global_steps,self.steer_plan)
            except Exception:
                if minimal(self.steer_plan):
                    from steer_opd.training.retention import recover_checkpoint
                    recover_checkpoint(checkpoint_root,self.steer_plan)
                raise
            if minimal(self.steer_plan):
                from steer_opd.training.retention import prune_checkpoints
                prune_checkpoints(checkpoint_root,self.global_steps,self.steer_plan,committed=receipt)
            from steer_opd.training.resume_handover import retire_resume_source
            retire_resume_source(self.steer_plan,self.global_steps)
            return result

        def _load_checkpoint(self):
            self.handover.require_student()
            if minimal(self.steer_plan):
                from steer_opd.training.retention import recover_checkpoint
                checkpoint_root=ROOT/self.steer_plan['output']/'checkpoints'
                saved=recover_checkpoint(checkpoint_root,self.steer_plan)
                if saved is None and self.steer_plan.get('resume_source'):
                    self.config.trainer.resume_mode='resume_path'
                    self.config.trainer.resume_from_path=str(ROOT/self.steer_plan['resume_source']['checkpoint'])
            with phase(self.log_folder,'checkpoint_load'):return super()._load_checkpoint()

    class SteerTaskRunner(BaseTaskRunner):
        def run(self,config):
            from omegaconf import OmegaConf
            from verl.utils.config import omega_conf_to_dataclass,validate_config
            from verl.workers.config import HFModelConfig
            from verl.trainer.ppo.utils import create_rl_dataset,create_rl_sampler
            from verl.utils.dataset.rl_dataset import collate_fn
            register_token_reward_direct()
            plan=OmegaConf.to_container(config.steer_opd,resolve=True)
            validate_training_sampling_policy(plan)
            correction=config.algorithm.get('rollout_correction')
            if (plan.get('student_logprob_source')!=STUDENT_LOGPROB_SOURCE or not correction
                    or not correction.get('bypass_mode') or correction.get('rollout_is') is not None
                    or correction.get('rollout_rs') is not None or correction.get('score_centering',False)
                    or not config.actor_rollout_ref.rollout.calculate_log_probs
                    or config.actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode!='raw_logprobs'):
                raise ValueError('OPD requires raw model logprobs and old-policy bypass; transformed sampling probabilities must not replace the raw loss/Top16 inputs')
            scorer=ScorerClient(plan)
            try:
                # Warm frozen engines first and sleep them before student model
                # initialization. All processes inherit the same two-card lease.
                scorer.start()
                _,worker_group=self.add_actor_rollout_worker(config)
                validate_config(config=config,use_reference_policy=False,use_critic=False)
                model=omega_conf_to_dataclass(config.actor_rollout_ref.model,dataclass_type=HFModelConfig)
                train=create_rl_dataset(config.data.train_files,config.data,model.tokenizer,model.processor,is_train=True)
                evaluation=create_rl_dataset(config.data.val_files,config.data,model.tokenizer,model.processor,is_train=False)
                trainer=SteerTrainer(config=config,tokenizer=model.tokenizer,processor=model.processor,
                    role_worker_mapping=self.role_worker_mapping,resource_pool_manager=self.init_resource_pool_mgr(config),
                    ray_worker_group_cls=worker_group,train_dataset=train,val_dataset=evaluation,
                    collate_fn=collate_fn,train_sampler=create_rl_sampler(config.data,train))
                trainer.steer_plan=plan;trainer.scorer=scorer
                trainer.init_workers()
                trainer.actor_rollout_wg.set_loss_fn(verl_loss)
                trainer.fit()
            finally:
                trainer=locals().get('trainer')
                if trainer is not None:
                    monitor=getattr(trainer,'overlap_monitor',None)
                    if monitor is not None:monitor.close()
                    writer=getattr(trainer.validation_generations_logger,'writer',None)
                    if writer is not None:writer.close()
                with phase(ROOT/plan['output']/'logs','frozen_scorer_process_shutdown'):scorer.close()
    return ray.remote(SteerTaskRunner)
