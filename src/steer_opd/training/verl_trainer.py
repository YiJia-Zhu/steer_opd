"""Pinned VERL trainer extension; all distributed execution remains upstream."""
from steer_opd.training.config import TrainingLossConfig
from steer_opd.training.verl_adapter import attach_advantages, verl_loss
from steer_opd.training.scoring import batch_rows
from steer_opd.training.scorer_client import ScorerClient


def task_runner_class():
    # Heavy framework imports are delayed until an explicitly accepted run.
    import ray
    from verl.trainer.main_ppo_v0 import BaseTaskRunner
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class SteerTrainer(RayPPOTrainer):
        def _update_actor(self,batch):
            # Upstream already recomputed current student old_log_probs, and no
            # optimizer step has occurred. Every call makes exactly one update.
            rows=batch_rows(batch)
            scored=self.scorer.score(rows)
            attach_advantages(batch,scored,TrainingLossConfig(**self.steer_plan['training_loss']))
            return super()._update_actor(batch)

    class SteerTaskRunner(BaseTaskRunner):
        def run(self,config):
            from omegaconf import OmegaConf
            from verl.utils.config import omega_conf_to_dataclass, validate_config
            from verl.workers.config import HFModelConfig
            from verl.trainer.ppo.utils import create_rl_dataset,create_rl_sampler
            from verl.utils.dataset.rl_dataset import collate_fn
            plan=OmegaConf.to_container(config.steer_opd,resolve=True)
            scorer=ScorerClient(plan)
            try:
                scorer.start()
                _,worker_group=self.add_actor_rollout_worker(config)
                validate_config(config=config,use_reference_policy=False,use_critic=False)
                model=omega_conf_to_dataclass(config.actor_rollout_ref.model,dataclass_type=HFModelConfig)
                train=create_rl_dataset(config.data.train_files,config.data,model.tokenizer,model.processor,is_train=True)
                confirm=create_rl_dataset(config.data.val_files,config.data,model.tokenizer,model.processor,is_train=False)
                trainer=SteerTrainer(config=config,tokenizer=model.tokenizer,processor=model.processor,
                    role_worker_mapping=self.role_worker_mapping,resource_pool_manager=self.init_resource_pool_mgr(config),
                    ray_worker_group_cls=worker_group,train_dataset=train,val_dataset=confirm,
                    collate_fn=collate_fn,train_sampler=create_rl_sampler(config.data,train))
                trainer.steer_plan=plan;trainer.scorer=scorer
                trainer.init_workers()
                # Public ONE_TO_ALL RPC installs the same callable on all FSDP
                # workers, retaining upstream packing and global-token reduction.
                trainer.actor_rollout_wg.set_loss_fn(verl_loss)
                trainer.fit()
            finally:
                scorer.close()
    return ray.remote(SteerTaskRunner)
