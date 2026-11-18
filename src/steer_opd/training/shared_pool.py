"""Explicit handover between VERL and frozen scorers on the same two GPUs."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

from steer_opd.experiment_runtime import timestamp


class SharedCheckpointManager:
    """Track the pinned trainer's blocking rollout sleep/weight-sync boundaries."""
    def __init__(self,manager,scorer,log_folder=None):
        self.manager=manager;self.scorer=scorer;self.rollout_asleep=True;self.log_folder=log_folder

    def __getattr__(self,key):return getattr(self.manager,key)

    def sleep_replicas(self):
        if not self.rollout_asleep:
            from steer_opd.training.telemetry import phase
            with phase(self.log_folder,'student_rollout_sleep'):self.manager.sleep_replicas()
            self.rollout_asleep=True

    def update_weights(self,*args,**kwargs):
        self.scorer.ensure_asleep()
        self.rollout_asleep=False  # A failed wake must not be mistaken for sleep.
        from steer_opd.training.telemetry import phase
        with phase(self.log_folder,'student_rollout_weight_sync'):return self.manager.update_weights(*args,**kwargs)


class SharedPoolHandover:
    def __init__(self,actor,checkpoints,scorer,event_path=None):
        self.actor=actor;self.checkpoints=checkpoints;self.scorer=scorer
        self.event_path=Path(event_path) if event_path else None
        self.phase='student'

    def event(self,name,**values):
        if self.event_path:
            self.event_path.parent.mkdir(parents=True,exist_ok=True)
            with self.event_path.open('a') as stream:
                stream.write(json.dumps({'time':timestamp(),'event':name,**values})+'\n')

    def require_student(self):
        if self.phase!='student':raise RuntimeError('Cannot enter student phase before frozen scorers sleep')
        self.scorer.ensure_asleep()

    @contextmanager
    def frozen_scoring(self):
        self.require_student()
        if not self.checkpoints.rollout_asleep:
            raise RuntimeError('Student rollout must acknowledge sleep before teacher scoring')
        started=time.monotonic()
        # ONE_TO_ALL is blocking: all FSDP ranks finish offloading before any
        # EasySteer role can wake on the shared physical cards.
        from steer_opd.training.telemetry import phase
        with phase(self.event_path.parent if self.event_path else None,'student_model_optimizer_offload'):
            self.actor.to('cpu',model=True,optimizer=True,grad=True)
        self.scorer.begin_shared_phase()
        self.phase='frozen_scoring';self.event('student_offload_committed')
        try:yield
        finally:
            # Failed/missing sleep ACK deliberately prevents a student update.
            self.scorer.end_shared_phase()
            self.phase='student'
            self.event('frozen_sleep_committed',seconds=time.monotonic()-started)
