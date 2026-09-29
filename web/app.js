'use strict';
const $=id=>document.getElementById(id);
let documents=[], activeEvidence=null;
const stateLabels={legacy_unverified:'Legacy · unverified',source_available:'Original PDF available',partial_text:'Partial text coverage'};
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function notify(text,error=false){$('notice').hidden=false;$('notice').textContent=text;$('notice').className=error?'error':'';}
async function api(path,body,options={}){const response=await fetch(path,{...options,method:options.method||(body===undefined?'GET':'POST'),headers:{'X-FinSight-Client':'local-ui',...(body===undefined?{}:{'Content-Type':'application/json'}),...options.headers},body:body===undefined?options.body:JSON.stringify(body)});if(!response.ok){let message;try{const data=await response.json();message=typeof data.detail==='string'?data.detail:JSON.stringify(data.detail);}catch{message=`Request failed (${response.status})`;}throw Error(message);}return response.json();}
async function busy(button,work){const text=button.textContent;button.disabled=true;button.textContent='Working…';try{await work();}catch(e){notify(e.message,true);}finally{button.disabled=false;button.textContent=text;}}
function view(name){document.querySelectorAll('.view').forEach(x=>x.hidden=x.id!==name);document.querySelectorAll('.nav').forEach(x=>x.classList.toggle('active',x.dataset.view===name));$('breadcrumb').textContent={research:'EVIDENCE RESEARCH',library:'FILING LIBRARY',reports:'COMPANY BRIEF'}[name];}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>view(b.dataset.view));
$('go-library').onclick=()=>view('library');
function options(select,items,placeholder){const current=select.value;select.replaceChildren();if(placeholder)select.append(new Option(placeholder,''));items.forEach(([value,label])=>select.append(new Option(label,value)));if(items.some(i=>i[0]===current))select.value=current;}
function updateFilings(){options($('document'),documents.filter(d=>!$('company').value||d.company===$('company').value).map(d=>[d.id,d.name]),'All filings');}
$('company').onchange=updateFilings;
async function refresh(){documents=await api('/api/documents');const s=await api('/api/status');options($('company'),s.companies.map(c=>[c,c]),'All companies');options($('report-company'),s.companies.map(c=>[c,c]));updateFilings();$('stats').replaceChildren();[[s.documents,'Registered documents'],[s.blocks.toLocaleString(),'Searchable evidence blocks'],[s.companies.length,'Issuers']].forEach(([v,t])=>{const d=node('div',undefined,'stat');d.append(node('b',String(v)),node('span',t));$('stats').append(d);});$('documents').replaceChildren();if(!documents.length)$('documents').append(node('p','No filings yet. Import an original PDF above.'));documents.forEach(d=>{const card=node('article',undefined,'card'),top=node('div',undefined,'card-top');top.append(node('strong',d.company),node('span',stateLabels[d.status]||d.status,'pill'));card.append(top,node('h3',d.name),node('p',`${d.kind} · ${d.filing_date||'Date unverified'} · ${d.page_count} PDF pages · ${d.blocks} evidence blocks`,'muted'));const b=node('button','Inspect coverage','secondary');b.onclick=()=>showDocument(d.id);card.append(b);$('documents').append(card);});const jobs=await api('/api/jobs');$('jobs').replaceChildren(...jobs.slice(0,10).map(j=>node('p',`${j.created_at} UTC · ${j.state} · ${j.message}`,'muted')));}
$('refresh').onclick=()=>busy($('refresh'),refresh);
$('legacy').onclick=()=>busy($('legacy'),async()=>{await api('/api/import-legacy',{});await refresh();notify('Legacy excerpts loaded. They remain explicitly unverified.');});
$('upload-form').onsubmit=e=>{e.preventDefault();const form=e.currentTarget;busy(form.querySelector('button'),async()=>{const data=new FormData(form),file=data.get('file');if(file.size>35*1024*1024)throw Error('PDF exceeds 35 MB.');const q=new URLSearchParams({company:data.get('company'),name:file.name,kind:data.get('kind')});if(data.get('filing_date'))q.set('filing_date',data.get('filing_date'));const r=await api('/api/documents/pdf?'+q,undefined,{method:'POST',body:file,headers:{'Content-Type':'application/pdf'}});await refresh();notify('Filing indexed. Inspect page coverage before using extracted financial values.');await showDocument(r.id);});};
function scope(){return {company:$('company').value||null,document_id:$('document').value||null,as_of:$('asof').value||null};}
function evidenceCard(e){const card=node('article',undefined,'card');const top=node('div',undefined,'card-top');top.append(node('strong',e.company||'Source excerpt'),node('span',`PDF p. ${e.page}`,'pill'));card.append(top,node('p',`${e.name||''} · ${e.filing_date||'Date unverified'} · ${stateLabels[e.status]||''}`,'muted'),node('pre',e.text,'excerpt'));const actions=node('div',undefined,'card-actions'),button=node('button','Inspect source ↗','secondary');button.onclick=()=>showSource(e.id);actions.append(button,node('small',`Evidence ${e.id.slice(0,8)}`));card.append(actions);return card;}
function nativeTable(item){
  const wrap=node('div',undefined,'native-table'),table=node('table');
  item.cells.forEach(cells=>{const row=node('tr');cells.forEach(value=>{
    const cell=node('td',value===null?'':value);cell.title=value===null?'No detected cell':value===''?'Empty source cell':value;
    if(value===null)cell.className='missing-cell';row.append(cell);
  });table.append(row);});
  wrap.append(table);return wrap;
}
$('search-form').onsubmit=e=>{e.preventDefault();busy(e.currentTarget.querySelector('button'),async()=>{
  const r=await api('/api/research',{question:$('question').value,...scope()});
  $('result-title').textContent=r.evidence.length?'Retrieved evidence':'No matching evidence';
  const mode=r.retrieval?.dense_status==='ready'?'Hybrid search':'Keyword search';
  $('result-meta').textContent=`${r.evidence.length} passages · ${r.latency_ms} ms · ${mode}`;
  $('results').replaceChildren(node('p',r.message,'warning'));
  if(r.retrieval?.dense_status==='stale_rebuild_required')$('results').append(node('p','The semantic index needs rebuilding after corpus changes. These results use keyword search.','warning'));
  if(r.corpus.unknown_date_excluded)$('results').append(node('p',`${r.corpus.unknown_date_excluded} filing(s) excluded because their dates are unknown.`,'warning'));
  r.evidence.forEach(e=>{
    $('results').append(evidenceCard(e));
    const context=r.source_context?.find(c=>c.anchor_id===e.id);
    if(context?.neighbors.length){
      const nearby=node('details');nearby.append(node('summary','Read nearby source passages'),node('p',context.limitation,'muted'));
      context.neighbors.forEach(n=>nearby.append(evidenceCard(n)));
      if(context.omitted_ids.length)nearby.append(node('p','Some nearby passages exceeded the context size limit. Inspect the original page.','warning'));
      $('results').append(nearby);
    }
    const layout=context?.structure_context;
    if(layout?.items?.length||layout?.omitted_ids?.length){
      const details=node('details');details.append(node('summary','Page headings and table candidates'),node('p',layout.limitation,'muted'));
      (layout.items||[]).forEach(item=>{
        if(item.kind==='native_table_candidate'){
          details.append(nativeTable(item));
          (item.margin_lines||[]).forEach(line=>details.append(node('pre',line.text,'excerpt')));
        }else details.append(node('pre',item.text,'excerpt'));
        const actions=node('div',undefined,'card-actions');
        item.evidence_ids.forEach(id=>{const button=node('button',`Inspect ${id.slice(0,8)}`,'text-button');button.onclick=()=>showSource(id);actions.append(button);});
        details.append(actions);
      });
      if(layout.omitted_ids?.length)details.append(node('p','Some native page context exceeded the size limit. Inspect the original page.','warning'));
      $('results').append(details);
    }
  });
  if(r.knowledge_context?.concepts.length){
    const details=node('details'),summary=node('summary','Research methodology · not company evidence');details.append(summary);
    r.knowledge_context.concepts.forEach(c=>{details.append(node('h3',c.metadata.title||c.id),node('p',`Declared review: ${c.declared_trust}. Review identity has not been authenticated.`,'muted'),node('pre',c.body,'excerpt'));});
    $('results').append(details);
  }
  const a=node('a','Export this search and evidence','button secondary');a.href='/api/runs/'+r.id;$('results').append(a);
});};
document.querySelectorAll('[data-query]').forEach(b=>b.onclick=()=>{$('question').value=b.dataset.query;$('search-form').requestSubmit();});
async function showSource(id){try{const e=await api('/api/evidence/'+id);activeEvidence=e;$('source-content').replaceChildren(node('p',`${e.company} · ${e.name} · PDF page ${e.page}`,'source-meta'),node('p',stateLabels[e.status],'pill'),node('pre',e.text,'source-text'));if(e.bbox){const a=node('a','Open original PDF ↗','button secondary');a.href=`/api/documents/${e.document_id}/pdf#page=${e.page}`;a.target='_blank';a.rel='noopener';const image=node('img',undefined,'source-image');image.alt=`Original PDF page ${e.page}, selected evidence outlined`;image.src=`/api/evidence/${id}/image`;image.onerror=()=>{image.replaceWith(node('p','Source rendering unavailable. Restore the original PDF to inspect this passage.','warning'));};$('source-content').append(a,image);}else $('source-content').append(node('p','Original PDF unavailable. This legacy excerpt cannot establish verified financial values.','warning'));if(!$('source-dialog').open)$('source-dialog').showModal();}catch(e){notify(e.message,true);}}
$('close-source').onclick=()=>$('source-dialog').close();
async function showDocument(id){
  const d=await api('/api/documents/'+id),content=$('source-content');
  content.replaceChildren(node('h3',d.name),node('p',`${d.company} · ${d.kind} · ${d.status}`,'source-meta'));
  JSON.parse(d.warnings).forEach(w=>content.append(node('p',w,'warning')));
  if(d.manifest)content.append(node('p',`SHA-256: ${d.sha256}\nParser: ${d.manifest.parser} ${d.manifest.parser_version}. Metadata: ${d.manifest.metadata_origin}.`,'source-meta'));
  const coverage=d.layout_coverage;
  content.append(node('p',`Native layout: ${coverage.indexed_pages} of ${coverage.total_pages} pages indexed. Parser candidates remain unreviewed.`,'muted'));
  const table=node('table'),head=node('tr');
  ['PDF page','PDF label','Native text','Blocks','Source','Layout'].forEach(t=>head.append(node('th',t)));table.append(head);
  for(const p of d.pages){
    const row=node('tr');[p.page,p.label||'—',p.status,p.block_count].forEach(t=>row.append(node('td',String(t))));
    const cell=node('td'),button=node('button','View','text-button');
    button.onclick=async()=>{try{const blocks=await api(`/api/documents/${id}/pages/${p.page}`);if(blocks.length)await showSource(blocks[0].id);else notify('This page has no native text. Inspect it in the original PDF.');}catch(e){notify(e.message,true);}};
    cell.append(button);row.append(cell);
    const layoutCell=node('td'),entry=coverage.pages.find(e=>e.page===p.page);
    if(entry?.status==='available'){
      const inspect=node('button',`${entry.tables} tables · ${entry.headings} headings`,'text-button');
      inspect.setAttribute('aria-label',`Inspect layout on page ${p.page}`);
      inspect.onclick=()=>busy(inspect,()=>showLayout(id,p.page));layoutCell.append(inspect);
    }else layoutCell.append(node('span',entry?'Rebuild required':'Not indexed','muted'));
    row.append(layoutCell);table.append(row);
  }
  if(d.pages.length){const wrap=node('div',undefined,'native-table');wrap.append(table);content.append(wrap);}
  if(!$('source-dialog').open)$('source-dialog').showModal();
}
async function showLayout(id,page){
  const result=await api(`/api/documents/${id}/pages/${page}/layout`),content=$('source-content');
  const back=node('button','Back to page coverage','secondary');back.onclick=()=>busy(back,()=>showDocument(id));
  content.replaceChildren(back,node('h3',`Native layout · PDF page ${page}`));
  if(result.status!=='available'){content.append(node('p',result.status==='stale_rebuild_required'?'Layout needs rebuilding.':'Native layout is unavailable for this page.','warning'));return;}
  const p=result.inventory;
  content.append(node('p',p.limitation,'warning'),node('p',`Source SHA-256: ${p.document_sha256}\nParser ${p.parser_version} · ${p.coordinates}`,'source-meta'));
  const original=node('a','Open original page','button secondary');original.href=`/api/documents/${id}/pdf#page=${page}`;original.target='_blank';original.rel='noopener';
  const download=node('a','Export page layout','button secondary');download.href=`/api/documents/${id}/pages/${page}/layout`;download.download=`layout-${id}-${page}.json`;
  const actions=node('div',undefined,'card-actions');actions.append(original,download);content.append(actions);
  content.append(node('h3',`Table candidates (${p.tables.length})`));
  if(!p.tables.length)content.append(node('p','No tables detected.','muted'));
  p.tables.forEach((t,i)=>{
    const details=node('details');details.open=true;
    details.append(node('summary',`Table ${i+1} · ${t.cells.length} rows · ${Math.max(0,...t.cells.map(r=>r.length))} columns`),nativeTable(t));
    (t.margin_lines||[]).forEach(line=>details.append(node('pre',line.text,'excerpt')));
    content.append(details);
  });
  const headings=node('details');headings.append(node('summary',`Heading candidates (${p.headings.length})`));
  p.headings.forEach(h=>headings.append(node('pre',h.text,'excerpt')));content.append(headings);
  const lines=node('details');lines.append(node('summary',`Native text lines (${p.lines.length})`));
  p.lines.forEach(line=>lines.append(node('pre',line.text,'excerpt')));content.append(lines);
  $('source-dialog').scrollTop=0;
}
$('build-report').onclick=()=>busy($('build-report'),async()=>{if(!$('report-company').value)throw Error('Import a company filing first.');const r=await api('/api/reports',{company:$('report-company').value});$('report-output').replaceChildren();const heading=node('div',undefined,'report-heading');heading.append(node('div','EXTRACTIVE RESEARCH PACK','eyebrow'),node('h1',r.title),node('p',r.limitation));$('report-output').append(heading);r.sections.forEach((s,i)=>{const section=node('section',undefined,'report-section');section.append(node('h2',`${String(i+1).padStart(2,'0')} / ${s.title}`));if(!s.evidence.length)section.append(node('p','No matching evidence retrieved for this section. This is a coverage gap.','warning'));s.evidence.forEach(e=>section.append(evidenceCard(e)));$('report-output').append(section);});$('print').disabled=false;$('export').hidden=false;$('export').href='/api/runs/'+r.id;});
$('print').onclick=()=>window.print();
refresh().catch(e=>notify(e.message,true));
