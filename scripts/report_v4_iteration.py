"""Derive the v4 follow-up tables and plots from immutable predictions."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mmso.backbone_study import digest
from mmso.frontier_evals import summarize
from mmso.reference_costs import openai_cost

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/v4-iteration-v2'


def read(path): return [json.loads(s) for s in path.read_text().splitlines()]


def paired(rows,left,right):
    left={p['id']:p for p in left};right={p['id']:p for p in right};groups=defaultdict(list)
    def score(pred,row):
        if pred.get('status')!='ok':return 0
        choice=pred['choice'] if 'choice' in pred else int(np.argmax(pred['probabilities']))
        return int(choice==row['target'])
    for row in rows:
        if row['id'] in left and row['id'] in right:
            groups[row['group_id']].append(score(right[row['id']],row)-score(left[row['id']],row))
    totals=np.asarray([(sum(v),len(v)) for v in groups.values()])
    sample=np.random.default_rng(20260925).integers(0,len(totals),(5000,len(totals)))
    draws=totals[sample].sum(1)
    return {'cases':int(totals[:,1].sum()),'right_minus_left_accuracy':float(totals[:,0].sum()/totals[:,1].sum()),
        'paired_group_bootstrap_95':np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist()}


def main():
    rows=read(ROOT/'data/v4-frontier/manifest.jsonl')
    expected={'mini-base':'frontier-base-v1','mini-adapted':'frontier-adapted-v1','qwen3-base':'frontier-qwen3-base-v1','gemma4-12b':'frontier-gemma4-12b-v1',
              'jev':'jev-mmlu-pro-v2','openai-sol-none':'openai-sol-none-v1','openai-astra-medium-pilot':'openai-astra-medium-v1'}
    predictions,models,sources={}, {}, {}
    for name,attempt in expected.items():
        path=REPORT/'attempts'/attempt/'summary.json'
        if not path.exists():continue
        raw=json.loads(path.read_text()); preds=read(path.parent/'predictions.jsonl')
        ids={p['id'] for p in preds}; selected=[r for r in rows if r['id'] in ids]
        predictions[name]=preds
        models[name]={'run_status':raw['status'],'cases':len(preds),'metrics':summarize(selected,preds),
            'source':str(path.relative_to(ROOT)), 'latency_scope':'API client including network' if name in ('jev','openai-sol-none','openai-astra-medium-pilot') else 'Local H100 pipeline, network excluded'}
        sources[str(path.relative_to(ROOT))]=digest(path)
        sources[str((path.parent/'predictions.jsonl').relative_to(ROOT))]=digest(path.parent/'predictions.jsonl')
    differences={}
    for benchmark in ('mmstar','mmau','mmlu_pro'):
        selected=[r for r in rows if r['benchmark']==benchmark]
        differences[benchmark]={}
        for other in ('mini-adapted','qwen3-base','gemma4-12b','jev','openai-sol-none'):
            if other not in predictions or benchmark not in models[other]['metrics']:continue
            differences[benchmark]['mini-base_to_'+other]=paired(selected,predictions['mini-base'],predictions[other])
    speed_rows=read(REPORT/'attempts/shared-observation-v1/shared-observation.jsonl')
    speed={}
    for count in (1,4,16):
        values=[r for r in speed_rows if r['questions']==count]
        modes={k:np.asarray([x for r in values for x in r['timings_ms'][k]]) for k in values[0]['timings_ms']}
        speed[str(count)]={'observations':len(values),'questions_compared':len(values)*count,
            'modes':{k:{'median_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95))} for k,v in modes.items()},
            'serial_media_reuse_speedup':float(np.median(modes['independent'])/np.median(modes['serial_shared_media'])),
            'packed_speedup':float(np.median(modes['independent'])/np.median(modes['packed_shared_media'])),
            'agreement':{mode:{'argmax':sum(r['comparison'][mode]['argmax_agreement'] for r in values),
                'max_probability_delta':max(r['comparison'][mode]['max_probability_delta'] for r in values)} for mode in ('serial_shared_media','packed_shared_media')},
            'reorder_max_probability_delta':max(r['reorder_max_probability_delta'] for r in values)}
    result={'models':models,'paired_differences':differences,'complete_request_speed':speed,'source_hashes':sources,
        'no_frontier_parity_established':True,'unchanged_served_model':'mmso-joint-v3',
        'quality_scope':'Public benchmark audit; not a fresh product acceptance set. Astra pilot has one question per stratum.'}
    api_costs={}
    for name in ('jev','openai-sol-none'):
        if name not in predictions:continue
        selected=[p for p in predictions[name] if p['id'].startswith('mmlu_pro:')]
        if name=='jev': total=sum(p.get('cost_usd',0) for p in selected)
        else: total=sum(openai_cost(p['usage'],{'input_usd_per_million':2,'output_usd_per_million':10})['standard_rate_estimate_usd'] for p in selected if p.get('usage'))
        correct=models[name]['metrics']['mmlu_pro']['correct']
        api_costs[name]={'cases':len(selected),'correct':correct,'inference_standard_rate_usd':total,
            'usd_per_1000_correct_decisions':total/correct*1000 if correct else None,
            'scope':'All requests in this matched text run, including incorrect decisions; standard API list-rate estimate before credits, excluding earlier setup/failed attempts'}
    result['matched_text_api_cost_per_correct']=api_costs
    audit_path=REPORT/'grounding-action-space-audit.json'
    if audit_path.exists():
        result['grounding_action_space_audit']=json.loads(audit_path.read_text())
        sources[str(audit_path.relative_to(ROOT))]=digest(audit_path)
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Generated v4 follow-up results','', 'Generated from the retained prediction files. Percentages below measure different task distributions; no pooled frontier score is computed.','',
        '## Full public audit','', '| Model | MMAU, 1,000 | MMStar, 1,500 | MMLU-Pro subset, 448 |','|---|---:|---:|---:|']
    for name in ('mini-base','mini-adapted','qwen3-base','gemma4-12b','jev','openai-sol-none'):
        if name not in models:continue
        cells=[]
        for benchmark in ('mmau','mmstar','mmlu_pro'):
            m=models[name]['metrics'].get(benchmark)
            cells.append(f"{m['accuracy']:.2%} ({m['correct']}/{m['cases']})" if m and (benchmark!='mmstar' or m['cases']==1500) else '—')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', 'A dash means not run on that full benchmark. For OpenAI visual comparisons, use the matched 120 cases below.','',
        '## Paired MMStar subset','', '| Model | Same visual cases | Accuracy | 95% grouped interval |','|---|---:|---:|---|']
    vision_ids={p['id'] for p in predictions.get('openai-sol-none',[]) if p['id'].startswith('mmstar:')}
    if vision_ids:
        subrows=[r for r in rows if r['id'] in vision_ids]
        for name in ('mini-base','mini-adapted','qwen3-base','gemma4-12b','openai-sol-none'):
            if name not in predictions:continue
            m=summarize(subrows,[p for p in predictions[name] if p['id'] in vision_ids])['mmstar']
            lo,hi=m['accuracy_group_bootstrap_95']
            lines.append(f"| {name} | {m['cases']} | {m['accuracy']:.2%} | {lo:.2%}–{hi:.2%} |")
    else:lines += ['| OpenAI comparison | pending | — | — |']
    lines += ['', '## Text decision latency','', 'The timing boundaries differ. Local H100 results exclude network/queue; provider references include network and hosted service work. These columns are not a claim of matched end-to-end speed.','',
        '| Model | MMLU-Pro accuracy | Median ms | p95 ms | Timing boundary |','|---|---:|---:|---:|---|']
    for name,m in models.items():
        if name=='openai-astra-medium-pilot':continue
        metric=m['metrics'].get('mmlu_pro'); latency=metric['latency_ms'] if metric else None
        if latency:
            lines.append(f"| {name} | {metric['accuracy']:.2%} | {latency['median']:.2f} | {latency['p95']:.2f} | {m['latency_scope']} |")
    lines += ['', '### Matched text API cost per correct decision','', '| Reference | All 448 requests, USD | USD per 1,000 correct decisions |','|---|---:|---:|']
    for name,cost in api_costs.items():
        lines.append(f"| {name} | {cost['inference_standard_rate_usd']:.6f} | {cost['usd_per_1000_correct_decisions']:.4f} |")
    lines += ['', 'Includes spending on wrong answers; uses returned token usage and standard list rates including cache writes. Earlier setup/failures remain in the overall spend ledger. MiSO has no measured hosted-serving cost per correct decision yet; training/study spend and busy-GPU proxies do not substitute for it.']
    lines += ['', '## Reasoning-enabled OpenAI pilot','', 'GPT-6 Astra medium has only 20 selected cases, one per text/visual category. This is a capability smoke check with a 1,024-output-token cap; inspect truncations before interpreting it.','']
    if 'openai-astra-medium-pilot' in models:
        for benchmark,m in models['openai-astra-medium-pilot']['metrics'].items():
            lines.append(f"- {benchmark}: {m['correct']}/{m['cases']} correct; completed-answer coverage {m['coverage']:.2%}; failures `{json.dumps(m['failures'])}`.")
    else:lines.append('Pending or unavailable; no score claimed.')
    lines += ['', '## Same-observation speed','', '| Questions | Independent ms | Default media reuse ms | Experimental packed ms | Default speedup | Packed speedup |','|---|---:|---:|---:|---:|---:|']
    for count,m in speed.items():
        times=m['modes']
        lines.append(f"| {count} | {times['independent']['median_ms']:.2f} | {times['serial_shared_media']['median_ms']:.2f} | {times['packed_shared_media']['median_ms']:.2f} | {m['serial_media_reuse_speedup']:.2f}× | {m['packed_speedup']:.2f}× |")
    lines += ['', 'Each cell pools five repeats on each of five observations, with paths interleaved. Default reuse matched all 105 probability vectors in this diagnostic. Packed mode preserved 105/105 top choices but differed by up to 0.0364 in probability.','', '## Fresh screen diagnostic','']
    screen_path=REPORT/'attempts/screen-confirmation-v1/summary.json'
    if screen_path.exists():
        screen=json.loads(screen_path.read_text())
        lines.append(f"On {screen['cases']} unused screenshots: **{screen['region_accuracy']:.2%} coarse region accuracy**.")
        audit=result.get('grounding_action_space_audit')
        if audit and not audit['can_discriminate_model_quality']:
            lines.append('**Correction: the earlier 0/117 click result cannot measure model quality.** None of the nine allowed grid-center points lies in any target box, so even an oracle would score zero. Original predictions are preserved. Precise GUI grounding remains unmeasured; see the [action-space audit](grounding-action-space-audit.json).')
        else:
            lines.append('Do not interpret the grid-center hit fraction as model grounding quality without checking the feasible-action ceiling. It is not an official full ScreenSpot-Pro score.')
    else:lines.append('Pending; nomination and screenshot identities are frozen.')
    lines += ['', '## Paired accuracy changes','', 'Right model minus original MiniCPM base, on identical cases. Bootstrap resamples image/audio/question groups. These are exploratory public-audit comparisons.','',
        '| Benchmark | Right model | Cases | Difference | 95% paired interval |','|---|---|---:|---:|---|']
    for benchmark,changes in differences.items():
        for key,d in changes.items():
            lo,hi=d['paired_group_bootstrap_95'];name=key.replace('mini-base_to_','')
            lines.append(f"| {benchmark} | {name} | {d['cases']} | {d['right_minus_left_accuracy']*100:+.2f} pp | {lo*100:+.2f} to {hi*100:+.2f} pp |")
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    assets=ROOT/'docs/assets';figures=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':'#f7f7ef','axes.facecolor':'#f7f7ef','savefig.facecolor':'#f7f7ef'})
    def save(fig,name,alt):
        for suffix in ('svg','png'):
            path=assets/(name+'.'+suffix);fig.savefig(path,dpi=170,bbox_inches='tight')
            if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        figures.append({'path':str((assets/(name+'.svg')).relative_to(ROOT)),'sha256':digest(assets/(name+'.svg')),'alt':alt,'model_results':True})
        plt.close(fig)
    order=[name for name in ('mini-base','mini-adapted','qwen3-base','gemma4-12b','jev','openai-sol-none') if name in models]
    paired_vision={p['id'] for p in predictions.get('openai-sol-none',[]) if p['id'].startswith('mmstar:')}
    titles=['MMAU · audio (1,000)',f'MMStar · vision ({len(paired_vision) or 1500:,})','MMLU-Pro · text (448)']
    benchmarks=['mmau','mmstar','mmlu_pro']
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),sharey=True)
    colors={'mini-base':'#8a9f92','mini-adapted':'#38584a','qwen3-base':'#ab713b','gemma4-12b':'#9e5848','jev':'#7e4d8d','openai-sol-none':'#386e99'}
    for ax,benchmark,title in zip(axs,benchmarks,titles):
        for i,name in enumerate(order):
            m=models[name]['metrics'].get(benchmark)
            if benchmark=='mmstar' and paired_vision and m is not None:
                subrows=[r for r in rows if r['id'] in paired_vision]
                subpreds=[p for p in predictions[name] if p['id'] in paired_vision]
                m=summarize(subrows,subpreds)['mmstar']
            if m is None:
                ax.text(i,4,'N/A',ha='center',color='#666666',fontsize=9);continue
            lo,hi=m['accuracy_group_bootstrap_95'];value=m['accuracy']*100
            ax.bar(i,value,color=colors[name],width=.72)
            ax.errorbar(i,value,yerr=[[value-lo*100],[hi*100-value]],fmt='none',color='#222222',capsize=3)
            ax.text(i,hi*100+3,f'{value:.1f}%',ha='center',fontsize=9)
        ax.set_title(title);ax.set_ylim(0,106);ax.set_xticks(range(len(order)),[n.replace('mini-','Mini ').replace('qwen3-base','Qwen3').replace('gemma4-12b','Gemma 4 12B').replace('openai-sol-none','Sol none').replace('jev','Jev') for n in order],rotation=35,ha='right')
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Accuracy (%)')
    fig.suptitle('Broader benchmarks expose the capability gap',fontsize=17,y=1.04)
    fig.text(.02,-.06,'H100 direct decisions; Jev and GPT-6 Sol are API references. Error bars: 95% group bootstrap.\nWithin each panel, every model uses the same cases. Full MMStar results remain in the report. No frontier-parity claim.',fontsize=10)
    fig.tight_layout();save(fig,'v4-frontier-audit','Public benchmark accuracy by modality, with confidence intervals and explicit missing provider modalities.')
    fig,ax=plt.subplots(figsize=(9,4.8))
    labels={'independent':'Independent requests','serial_shared_media':'Reuse media · default','packed_shared_media':'Reuse + packed branches · experimental'}
    for mode,color in zip(labels,('#a08066','#38584a','#467ba0')):
        vals=[speed[str(n)]['modes'][mode]['median_ms'] for n in (1,4,16)]
        ax.plot([1,4,16],vals,'o-',label=labels[mode],color=color,linewidth=2)
        ax.annotate(f'{vals[-1]:.0f} ms',(16,vals[-1]),xytext=(7,0),textcoords='offset points',va='center')
    ax.set(xlabel='Questions about the same image + audio',ylabel='Complete local request median (ms)',xticks=[1,4,16],xlim=(.5,18.7),ylim=(0,2120))
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper left')
    ax.set_title('Encode the observation once',fontsize=17)
    fig.text(.08,-.055,'Five observations × five repetitions on one H100; includes media decoding, processing and encoders.\nDefault reuse matched all 105 tested probability vectors. Packed mode changed probabilities by up to 3.64 pp.\nNetwork, queue and cold model load excluded. This is a speed/equivalence diagnostic, not a new accuracy score.',fontsize=9)
    fig.tight_layout();save(fig,'v4-complete-request-speed','Complete local request latency as question count increases; default media reuse preserves tested probabilities.')
    (REPORT/'figures.json').write_text(json.dumps({'figures':figures,'source_summary_sha256':digest(REPORT/'summary.json')},indent=2)+'\n')
    print(json.dumps({'models':list(models),'default_speedup_16q':speed['16']['serial_media_reuse_speedup'],'packed_speedup_16q':speed['16']['packed_speedup']}))


if __name__=='__main__':main()
