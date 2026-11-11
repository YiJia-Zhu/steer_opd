"""Check core and related papers without network access or model execution."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / 'audit/package_validation.json'
EXPECTED = {'2610.40998v1', '2604.13016v2'}
RELATED = {
    '2609.38342v1', '2610.34005v1', '2610.31438v1', '2610.14022v1', '2610.27012v1',
    '2608.06802v1', '2609.10154v1', '2611.03133v1', '2609.38360v1',
}
ARCHIVED = {'2402.12842v3', '2610.10447v1'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_file = ROOT / 'papers/manifest.json'
    manifest = json.loads(manifest_file.read_text())
    assert {p['version'] for p in manifest['papers']} == EXPECTED
    assert len(manifest['papers']) == 2
    pdfs = sorted((ROOT / 'papers').glob('*.pdf'))
    assert {str(p.relative_to(ROOT)) for p in pdfs} == {p['pdf_file'] for p in manifest['papers']}
    related_file = ROOT / 'ref/manifest.json'
    related = json.loads(related_file.read_text())
    assert {p['version'] for p in related['papers']} == RELATED
    assert len(related['papers']) == len(RELATED)
    assert all(p['status'] == 'ready' for p in related['papers'])
    related_pdfs = sorted((ROOT / 'ref').glob('*.pdf'))
    assert {str(p.relative_to(ROOT)) for p in related_pdfs} == {p['pdf_file'] for p in related['papers']}
    archived_file = ROOT / 'ref/archive/manifest.json'
    archived = json.loads(archived_file.read_text())
    assert {p['version'] for p in archived['papers']} == ARCHIVED
    assert all(p['active_comparison'] is False for p in archived['papers'])
    archived_pdfs = sorted((ROOT / 'ref/archive').glob('*.pdf'))
    assert {str(p.relative_to(ROOT)) for p in archived_pdfs} == {p['pdf_file'] for p in archived['papers']}
    all_papers = manifest['papers'] + related['papers'] + archived['papers']
    for paper in all_papers:
        pdf, source, text = [ROOT / paper[key] for key in ['pdf_file', 'source_html', 'source_text']]
        assert pdf.read_bytes().startswith(b'%PDF-'), pdf
        assert sha256(pdf) == paper['pdf_sha256'], pdf
        assert sha256(source) == paper['source_html_sha256'], source
        assert sha256(text) == paper['source_text_sha256'], text
        first_page = text.parent / (paper['version'] + '.pdf_first_page.txt')
        assert paper['version'] in ''.join(first_page.read_text().split()), paper['version']
        for locator in paper.get('evidence_locators', []):
            assert locator['marker'].casefold() in text.read_text().casefold(), (paper['version'], locator['marker'])

    docs = sorted((ROOT / 'docs').glob('*.md'))
    required_docs = {'README.md', '01_research_plan.md', '02_paper_settings.md',
                     '03_execution_plan.md',
                     '09_novelty_audit_cross_family_steering_opd.md'}
    assert required_docs <= {path.name for path in docs}
    md_files = [ROOT / 'README.md', ROOT / 'AGENTS.md', ROOT / 'runs/README.md', *docs,
                ROOT / 'experiments/run_record_template.md', ROOT / 'ref/README.md',
                ROOT / 'ref/archive/README.md']
    local_count, missing, outside, arxiv_versions = 0, [], [], set()
    for path in md_files:
        content = path.read_text()
        assert content.count('```') % 2 == 0, path
        assert content.count('$$') % 2 == 0, path
        arxiv_versions.update(re.findall(r'arxiv\.org/(?:abs|pdf)/([0-9.]+v\d+)', content))
        for url in re.findall(r'\[[^\]]+\]\(([^)]+)\)', content):
            if url.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            target = (path.parent / url.split('#')[0]).resolve()
            local_count += 1
            if not target.is_relative_to(ROOT):
                outside.append({'file': str(path.relative_to(ROOT)), 'target': url})
            elif not target.exists() and target != REPORT:
                missing.append({'file': str(path.relative_to(ROOT)), 'target': url})
    assert not missing, missing
    assert not outside, outside
    novelty_root = ROOT / 'audit/novelty_search_2026-11-07'
    novelty_file = novelty_root / 'assessment.json'
    novelty = json.loads(novelty_file.read_text())
    novelty_sources_file = novelty_root / 'source_manifest.json'
    novelty_sources = json.loads(novelty_sources_file.read_text())['sources']
    novelty_versions = {p['version'] for p in novelty_sources}
    assert {p['version'] for p in novelty['papers']} == novelty_versions
    assert len(novelty_versions) == len(novelty_sources) == novelty['additional_fulltext_sources']
    assert EXPECTED | RELATED <= arxiv_versions
    assert arxiv_versions <= EXPECTED | RELATED | novelty_versions, arxiv_versions
    for source in novelty_sources:
        assert source['status'] == 200
        for kind in ['html', 'text', 'pdf']:
            if kind + '_file' in source:
                assert sha256(ROOT / source[kind + '_file']) == source[kind + '_sha256']
    selected_metadata_file = novelty_root / 'selected_metadata.json'
    selected_metadata = json.loads(selected_metadata_file.read_text())
    assert selected_metadata['status'] == 200
    assert {p['version'] for p in selected_metadata['records']} == novelty_versions
    for query in novelty['arxiv_queries']:
        data = json.loads((novelty_root / query['file']).read_text())
        assert data['status'] == 200 and len(data['records']) == query['returned']
    with (novelty_root / 'screening.csv').open(newline='') as stream:
        screening = list(csv.DictReader(stream))
    pool = json.loads((novelty_root / 'search_pool.json').read_text())
    assert len(screening) == len(pool) == novelty['unique_metadata_candidates']
    assert sum(novelty['evidence_depth_counts'].values()) == len(novelty_sources)

    notes = (ROOT / 'docs/02_paper_settings.md').read_text()
    templates = re.findall(r'```text\n(.*?)\n```', notes, re.S)
    rethink_text = (ROOT / 'papers/sources/2604.13016v2.txt').read_text()
    assert len(templates) == 2
    assert all(template in rethink_text for template in templates)

    matrix_file = ROOT / 'experiments/experiment_matrix.csv'
    with matrix_file.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    plan_file = ROOT / 'experiments/paper_aligned_selection_20261110.json'
    plan = json.loads(plan_file.read_text())
    group_ids = {row['id'] for row in rows}
    assert rows and len(group_ids) == len(rows)
    required_groups = set(plan['required_groups'])
    assert {row['id'] for row in rows if row['priority'] == 'required'} == required_groups
    assert all(row['primary_evaluation'] == 'student_without_hook' for row in rows)
    assert all(row['rollout_hook'] == 'false' for row in rows)
    assert all(row['status'] == 'planned' for row in rows)
    assert plan['executable_launch_config'] is False
    if plan['state'] == 'planning_only_paused':
        assert plan['gpu_execution_authorized'] is False
        assert plan['asset_download_authorized'] is False

    download_file = ROOT / 'experiments/download_manifest.json'
    downloads = json.loads(download_file.read_text())
    asset_keys = {a['key'] for a in downloads['assets']}
    assert len(asset_keys) == len(downloads['assets']) == 24
    assert len(downloads['profiles']) == 9
    for asset in downloads['assets']:
        assert re.fullmatch(r'[a-f0-9]{40}', asset['revision']), asset['key']
        record = json.loads((ROOT / asset['metadata_file']).read_text())
        assert record['status'] == 200 and record['data']['sha'] == asset['revision']
        assert record['data']['id'] == asset['repo_id'], asset['key']
        assert asset['allow_patterns']
        if asset['repo_type'] == 'model':
            assert asset['weight_bytes'] > 0
    for profile in downloads['profiles'].values():
        assert set(profile['assets']) <= asset_keys
    stage_file = ROOT / 'experiments/stage_matrix.csv'
    with stage_file.open(newline='') as stream:
        stage_rows = list(csv.DictReader(stream))
    assert stage_rows
    assert {row['group'] for row in stage_rows} <= group_ids
    assert {row['pair'] for row in stage_rows} <= {panel['id'] for panel in plan['panels']}
    assert len({(row['stage'], row['pair'], row['group']) for row in stage_rows}) == len(stage_rows)
    assert all(0 <= int(row['auxiliary_updates']) <= int(row['student_updates']) for row in stage_rows)
    assert all(row['status'] == 'planned' and row['launcher_implemented'] == 'false' for row in stage_rows)
    assert all(row['primary_evaluation'] == 'student_without_hook' for row in stage_rows)
    evidence = ROOT / 'audit/experiment_planning_2026-11-07'
    code_sources = json.loads((evidence / 'code_sources.json').read_text())
    for source in code_sources['files']:
        assert source['status'] == 200
        assert len((evidence / source['local_file']).read_bytes()) == source['bytes']

    cleanup_file = ROOT / 'audit/docs_cleanup_2026-11-10/manifest.json'
    cleanup = json.loads(cleanup_file.read_text())
    assert cleanup['deleted_documents'] == sum(
        row['action'].startswith('replace_') for row in cleanup['previous_documents'])
    for row in cleanup['previous_documents']:
        path = ROOT / row['path']
        if row['action'].startswith('replace_'):
            assert not path.exists(), f'Superseded exploration document recreated: {path}'
        else:
            assert sha256(path) == row['sha256'], path
        if row.get('concurrent_latest_snapshot'):
            assert sha256(ROOT / row['concurrent_latest_snapshot']) == row['pre_deletion_sha256']

    artifacts = [manifest_file, related_file, archived_file, matrix_file, download_file, stage_file,
                 plan_file, cleanup_file,
                 ROOT / 'scripts/download_assets.py', ROOT / 'scripts/validate_package.py',
                 novelty_file, novelty_sources_file, selected_metadata_file,
                 novelty_root / 'screening.csv', *md_files]
    result = {
        'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'root': str(ROOT), 'papers': len(all_papers), 'core_papers': len(pdfs),
        'related_papers': len(related_pdfs), 'archived_papers': len(archived_pdfs),
        'active_papers': len(pdfs) + len(related_pdfs),
        'pdf_hashes_verified': len(all_papers),
        'pdf_first_page_versions_verified': len(all_papers),
        'primary_source_snapshots_verified': len(all_papers),
        'experiment_documents': len(docs), 'markdown_files_checked': len(md_files),
        'local_links_checked': local_count, 'missing_local_links': missing,
        'links_outside_package': outside, 'exact_prompt_templates_verified': len(templates),
        'planned_experiment_groups': len(rows), 'required_experiment_groups': len(required_groups),
        'planned_stage_group_rows': len(stage_rows),
        'planning_state': plan['state'], 'active_model_panels': len(plan['panels']),
        'historical_download_assets': len(asset_keys),
        'historical_download_profiles': len(downloads['profiles']),
        'removed_superseded_documents': cleanup['deleted_documents'],
        'code_source_snapshots_verified': len(code_sources['files']),
        'novelty_search_queries': len(novelty['arxiv_queries']),
        'novelty_metadata_candidates': len(pool),
        'novelty_added_primary_texts_verified': len(novelty_sources),
        'novelty_evidence_depth_counts': novelty['evidence_depth_counts'],
        'artifacts': [{'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)} for path in artifacts],
        'scope': 'Material integrity and internal-reference checks only. No model training or experimental validation.',
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
