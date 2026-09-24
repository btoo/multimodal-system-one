"""Render measured pilot results exclusively from committed report/prediction files."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmso.artifacts import ROOT, read_manifest, sha256, write_json
from mmso.metrics import categorical_metrics

OUT=ROOT/'reports/pilot-v1'
OUT.mkdir(parents=True,exist_ok=True)
RUNS=['speech-cnn-v1','sounds-cnn-v1','screens-clip-v1']
reports=[json.loads((ROOT/'reports'/r/'report.json').read_text()) for r in RUNS]
speech,sounds,screen=reports
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'mmso-results-v1'})
BG='#f8fafc';INK='#14283e';TEAL='#007e78';GOLD='#b26614';BLUE='#2e62bc'


def save(fig,name):
    for extension in ['svg','png']:
        fig.savefig(OUT/f'{name}.{extension}',dpi=160,facecolor=BG,metadata={'Date':None} if extension=='svg' else None)
    p=OUT/f'{name}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    plt.close(fig)


fig,axs=plt.subplots(1,3,figsize=(14,5.8),facecolor=BG)
fig.subplots_adjust(top=.66,bottom=.22,wspace=.35,left=.07,right=.97)
fig.text(.06,.91,'First measured baselines',fontsize=24,weight='bold',color=INK)
fig.text(.06,.83,'Real-data implementation controls · one training seed · different tasks, separate scorecards',fontsize=11,color=INK)
for ax,r,title in zip(axs[:2],[speech,sounds],['8 spoken keywords','10 environmental sounds']):
    vals=[r['test_prior']['accuracy']*100,r['test_raw']['accuracy']*100]
    ax.set_facecolor(BG);ax.bar(['Class prior','Audio CNN'],vals,color=['#b5c3d2',TEAL],width=.6)
    for i,v in enumerate(vals):
        height=r['accuracy_interval']['high']*100 if i==1 else v
        ax.text(i,height+3,f'{v:.2f}%',ha='center',fontsize=13,weight='bold')
    ax.set_title(f"{title}\n{r['test_raw']['examples']} held-out records",fontsize=13,pad=16)
    interval=r['accuracy_interval'];acc=vals[1]
    ax.errorbar([1],[acc],yerr=[[acc-interval['low']*100],[interval['high']*100-acc]],fmt='none',color=INK,capsize=5)
    ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.spines[['top','right']].set_visible(False)
ax=axs[2];ax.set_facecolor(BG)
vals=[screen['oracle_crop_type']['accuracy']*100,screen['clip_grid']['hit_rate']*100]
ax.bar(['Oracle crop\ntext / icon','Coarse-grid\ngrounding'],vals,color=[BLUE,GOLD],width=.6)
for i,v in enumerate(vals):ax.text(i,v+3,f'{v:.2f}%',ha='center',fontsize=13,weight='bold')
ax.set_title('Screen diagnostics\n24 examples, six applications',fontsize=13,pad=16)
ax.set_ylim(0,105);ax.set_ylabel('Task success (%)');ax.spines[['top','right']].set_visible(False)
fig.text(.06,.07,'Audio bars: speaker/source-recording cluster intervals for a fixed checkpoint. Supplied UI crops do not demonstrate grounding.',fontsize=10,color=INK)
save(fig,'results')

fig,axs=plt.subplots(1,2,figsize=(12,5.8),facecolor=BG)
fig.subplots_adjust(top=.76,bottom=.19,left=.08,right=.97,wspace=.3)
fig.text(.06,.91,'Confidence on the held-out audio records',fontsize=21,weight='bold',color=INK)
fig.text(.06,.84,'Measured reliability; sparse bins and one seed limit the conclusions',fontsize=11,color=INK)
for ax,r,title in zip(axs,reports[:2],['Speech keywords','Environmental sounds']):
    ax.set_facecolor(BG);ax.plot([0,1],[0,1],'--',color='#90a0af',label='Ideal')
    for key,label,c in [('test_raw','Raw',GOLD),('test_calibrated','Temperature adjusted',TEAL)]:
        bins=[b for b in r[key]['reliability'] if b['count']]
        ax.plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',color=c,label=label)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence in bin',ylabel='Accuracy in bin',title=title)
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.text(.06,.06,'Temperatures were fitted on separate calibration data. Full bin counts, NLL, Brier, and risk/coverage are in each report.',fontsize=10,color=INK)
save(fig,'reliability')

# Preserve the preregistered row-weighted result; disclose within-split duplicates
# and compute a labeled post-hoc sensitivity from saved outputs without refitting.
rows=read_manifest(ROOT/'evals/manifests/speech_keywords.jsonl')
predictions={r['id']:r for r in read_manifest(ROOT/'reports/speech-cnn-v1/predictions.jsonl')}
seen=set();unique=[];removed=[]
for r in rows:
    if r['split']!='test':continue
    key=r['media'][0]['sha256']
    if key in seen:removed.append(r['id'])
    else:seen.add(key);unique.append(r)
labels=unique[0]['labels']
sensitivity=categorical_metrics(np.array([labels.index(r['target']) for r in unique]),[predictions[r['id']]['raw_probabilities'] for r in unique],labels)
summary={'status':'measured_baseline_pilot','runs':{r:sha256(ROOT/'reports'/r/'report.json') for r in RUNS},
         'official_benchmark_submission':False,'native_joint_model_trained':False,
         'speech_unique_media_sensitivity':{'post_hoc':True,'refit':False,'retained_examples':len(unique),'removed_duplicate_ids':removed,
                                           'accuracy':sensitivity['accuracy'],'nll':sensitivity['nll']},
         'note':'One exact duplicate pair is retained in the original speech test and one in calibration. Both pairs remain within a speaker and split; no train/test leakage. Original results are unchanged.'}
write_json(OUT/'summary.json',summary)

lines=['# First measured real-data baselines','',
       'These are implementation controls on small, pinned subsets. They are not an architecture search, official benchmark submission, or proof of native audio–screen reasoning.','',
       '![Measured baseline results](results.svg)','',
       '| Task | Held-out records | Accuracy | Class-prior accuracy | Raw → adjusted NLL | Warm p95 processing |',
       '|---|---:|---:|---:|---:|---:|']
for r in reports[:2]:
    title='Eight spoken keywords' if r['dataset']=='speech_keywords' else 'Ten environmental sounds'
    lines.append(f"| {title} | {r['test_raw']['examples']} | {r['test_raw']['accuracy']:.2%} | {r['test_prior']['accuracy']:.2%} | {r['test_raw']['nll']:.3f} → {r['test_calibrated']['nll']:.3f} | {r['timing']['inference']['warm_p95_ms']:.2f} ms |")
lines += ['',f"Speech uses {speech['parameters']:,} parameters; sound uses {sounds['parameters']:,}. Both were trained from scratch on MPS. The fixed recipe allowed at most 16 epochs / 120 synchronized training seconds per dataset; both reached the epoch limit first. The best development-NLL checkpoint was frozen before calibration and test evaluation.",'',
          'The speech split is speaker-disjoint. Sound preserves source-recording fold groups with a custom train/development/calibration/test assignment. Each report includes source and manifest hashes, checkpoint hashes, parameter counts, per-class results, all development epochs, and memory measurements.','',
          'Audio timing includes warm file decoding, resampling, feature extraction, inference, probability conversion, and synchronization. It excludes the time needed to listen to the approximately one-second speech or five-second sound clip. Endpointing, streaming, cold-process timing, and real-time false activations are not measured.','',
          '## Screen findings','',
          f"The generic CLIP control classified **{round(screen['oracle_crop_type']['accuracy']*24)}/24 supplied target crops** as text or icon. These are oracle crops: their location was supplied. This result does not show that the model can find the target itself.",'',
          f"Both center-point and coarse-grid grounding hit **0/24** targets. Only **{screen['grid_point_coverage_ceiling']['hits']}/24** target boxes contained any grid-center candidate at all. A perfect tile ranker therefore could not make this proposal scheme useful on the probe. Native-resolution element proposals or another spatial grounding method are needed before further score optimization.",'',
          f"CLIP-grid grounding had a {screen['timing']['grounding_pipeline_p95_ms']:.0f} ms warm p95 including image decoding, tiling, preprocessing, scoring, and synchronization. The model has {screen['model']['parameters']:,} frozen pretrained parameters. The two audio CNNs and this CLIP control are separate models; they are not a joint network.",'',
          '## Probability and data audit','',
          '![Measured audio reliability](reliability.svg)','',
          f"One exact duplicate pair occurs inside speech calibration and another inside speech test. Both are within the same speaker/split, and no media hash crosses splits. The original frozen result is retained. A post-hoc unique-media sensitivity over **{len(unique)}** test recordings gives **{sensitivity['accuracy']:.1%}** accuracy, without retraining or refitting temperature. See [the summary](summary.json).",'',
          'Temperature adjustment made small changes on these held-out records. The sound model still has material calibration error. No shifted-app, long-form speech, overlapping-event, or paired-input calibration claim follows from this pilot.','',
          '## Reproduction and evidence','',
          '- [Frozen pilot protocol](../../evals/pilot-protocol.json)',
          '- [Speech report](../speech-cnn-v1/report.json) and [predictions](../speech-cnn-v1/predictions.jsonl)',
          '- [Sound report](../sounds-cnn-v1/report.json) and [predictions](../sounds-cnn-v1/predictions.jsonl)',
          '- [Screen report](../screens-clip-v1/report.json) and [predictions](../screens-clip-v1/predictions.jsonl)',
          '- [Speech checkpoint](../../artifacts/speech-cnn-v1/model.safetensors) and [sound checkpoint](../../artifacts/sounds-cnn-v1/model.safetensors)','',
          'The baseline implementation revision is recorded in each report. Later generic scoring/reporting additions do not alter the saved outputs. Use a new run ID when reproducing training; never overwrite an existing run. Numerical and latency results can vary with hardware and library versions.','',
          '## Next experiment','',
          'Keep this screen probe as disclosed development evidence for any revised proposal method, and nominate a fresh, untouched screen sample for confirmation. Build a spoken-intent baseline and semantically paired human-audio/screen data before training native fusion. The current keyword and sound controls cannot substitute for those tasks.','']
(OUT/'README.md').write_text('\n'.join(lines))
print(f'Rendered measured results in {OUT}')
