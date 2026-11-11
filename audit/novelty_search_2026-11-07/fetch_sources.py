from pathlib import Path
import requests,json,re,datetime,concurrent.futures,hashlib,shutil
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
OLD=Path('/home/zhenning/data/working_tmp/idea/opd/sources')
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.skip=0
 def handle_starttag(self,tag,attrs):
  if tag in ('script','style'):self.skip+=1
  if tag in ('p','div','section','h1','h2','h3','h4','li','tr','br'):self.parts.append('\n')
 def handle_endtag(self,tag):
  if tag in ('script','style'):self.skip=max(0,self.skip-1)
 def handle_data(self,data):
  if not self.skip:self.parts.append(data)
 def text(self):return re.sub(r'\n\s*\n+','\n',re.sub(r'[^\S\n]+',' ',''.join(self.parts))).strip()
IDS=['2606.09456v1','2607.22334v1','2605.07711v3','2610.13257v1','2610.08448v1','2609.34738v1','2609.33816v1','2610.27555v1','2606.06021v4','2610.23306v1','2610.02381v1','2610.26397v2','2606.02530v2','2605.12400v2','2610.14001v1','2609.12579v1','2609.36484v2','2605.22675v2','2606.29340v1','2608.13040v1','2611.08016v1','2611.07241v1','2611.07731v1','2605.13230v2','2607.13399v1','2609.33455v2','2606.07082v3']
def fetch(version):
 dest=ROOT/'sources';dest.mkdir(exist_ok=True)
 meta={'version':version,'url':'https://arxiv.org/html/'+version,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  # Refresh official full text for this audit, even where an older local copy exists.
  r=requests.get(meta['url'],timeout=55);meta['status']=r.status_code;r.raise_for_status();p=Parser();p.feed(r.text);text=p.text()
  (dest/(version+'.html')).write_text(r.text);(dest/(version+'.txt')).write_text(text)
  meta.update(html_file=str((dest/(version+'.html')).relative_to(PROJECT)),text_file=str((dest/(version+'.txt')).relative_to(PROJECT)),html_sha256=hashlib.sha256(r.content).hexdigest(),text_sha256=hashlib.sha256(text.encode()).hexdigest(),chars=len(text))
 except Exception as e:meta['error']=str(e)
 (dest/(version+'.meta.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 return meta
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  out=[]
  for m in ex.map(fetch,IDS):out.append(m);print(m['version'],m.get('status'),m.get('chars'),m.get('error',''),flush=True)
 (ROOT/'source_manifest.json').write_text(json.dumps({'sources':out},ensure_ascii=False,indent=2)+'\n')
