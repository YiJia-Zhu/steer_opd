"""Read-only training observer; stop observing after three committed updates."""
from pathlib import Path
import datetime
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/step_4_train/ablation/T_Qwen3.5-35B-A3B_S_Qwen3-4B/OpenThoughts3-1.2M/shared/opd_n1024_project_outputs_utf8_v2'


def main():
    deadline = time.monotonic() + 2400
    offset, buffer = 0, ''
    updates, syncs, failures = [], 0, []
    last = None
    state = {'status': 'observing', 'updates': updates, 'failures': failures}
    while time.monotonic() < deadline:
        path = RUN / 'logs/training_events.jsonl'
        if not path.exists():
            time.sleep(5)
            continue
        with path.open() as stream:
            stream.seek(offset)
            buffer += stream.read()
            offset = stream.tell()
        lines = buffer.split('\n')
        buffer = lines.pop()
        for line in lines:
            if not line:
                continue
            event = json.loads(line)
            kind = event['event']
            if kind == 'optimizer_update_committed':
                updates.append(event)
                metrics = event.get('metrics', {})
                print('UPDATE', event['step'], event['time'],
                      {k:v for k,v in metrics.items() if k.startswith('alignment/')}, flush=True)
            if updates and kind == 'phase_end' and event.get('phase') == 'student_rollout_weight_sync' and event.get('status') == 'completed':
                syncs += 1
            if kind == 'phase_end' and event.get('status') == 'failed' or kind == 'child_exit':
                failures.append(event)
            if kind in ('phase_start', 'phase_end'):
                last = {k:event[k] for k in ('time', 'phase', 'status', 'step') if k in event}
        state.update(observed_at=datetime.datetime.now().astimezone().isoformat(),
                     committed_steps=[event['step'] for event in updates],
                     synchronized_updates=syncs, current_phase=last)
        if failures:
            state['status'] = 'failed'
        elif len(updates) >= 3 and syncs >= 3:
            plan = json.loads((RUN / 'metadata/plan.json').read_text())
            assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
                       for name, sha in plan['implementation_sha256'].items())
            accepted = json.loads((RUN / 'metadata/formal_runtime_acceptance.json').read_text())
            assert accepted['status'] == 'passed' and accepted['plan_binding'] == plan['binding']
            state.update(status='passed_previous_failure_step', plan_binding=plan['binding'],
                         implementation_matches_frozen_plan=True,
                         formal_runtime_acceptance=accepted)
        temporary = FOLDER / 'runtime_verification.json.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(FOLDER / 'runtime_verification.json')
        if state['status'] != 'observing':
            print('FINAL_STATUS', state['status'], state['committed_steps'], flush=True)
            return
        time.sleep(10)
    raise TimeoutError('Observer deadline reached; training was left untouched')


if __name__ == '__main__':
    main()
