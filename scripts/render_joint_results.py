"""Measured joint-model figures. Requires the nominated model and both controls evaluated."""
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmso.artifacts import ROOT,read_manifest,sha256,write_json
from mmso.joint_data import MANIFEST,expand_scenes
from mmso.joint_world import JOINT_TASKS
from mmso.metrics import aligned_predictions

RUNS=['joint-full-v2','joint-audio-only-v2','joint-image-only-v2']
evaluations={r:json.loads((ROOT/'reports'/r/'evaluation.json').read_text()) for r in RUNS}
training={r:json.loads((ROOT/'reports'/r/'training.json').read_text()) for r in ['joint-full-v1',*RUNS]}
full=evaluations[RUNS[0]]
OUT=ROOT/'reports/joint-v2';OUT.mkdir(parents=True,exist_ok=True)
BG='#f8fafc';INK='#14283e';TEAL='#007e78';GOLD='#b26614';BLUE='#2e62bc'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'joint-v2'})


def save(fig,name):
    for ext in ['svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=160,facecolor=BG,metadata={'Date':None} if ext=='svg' else None)
    path=OUT/f'{name}.svg';path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n');plt.close(fig)


fig,axes=plt.subplots(1,2,figsize=(14,6),facecolor=BG)
fig.subplots_adjust(left=.075,right=.97,top=.69,bottom=.22,wspace=.3)
fig.text(.065,.91,'The first unified decision model',fontsize=24,weight='bold',color=INK)
fig.text(.065,.83,'Real keyword audio + generated panels · six joint question families · one training seed',fontsize=11,color=INK)
for name,label,color in [('joint-full-v1','Fixed training pairs',GOLD),('joint-full-v2','Re-paired each batch',TEAL)]:
    history=training[name]['history'];axes[0].plot([h['steps'] for h in history],[h['joint_macro_accuracy']*100 for h in history],color=color,lw=2.5,label=label)
axes[0].set(title='Development: repair a data shortcut',xlabel='Optimizer steps',ylabel='Joint accuracy (%)',ylim=(35,100));axes[0].legend(frameon=False,fontsize=10)
labels=['Audio only','Image only','Joint model'];names=[RUNS[1],RUNS[2],RUNS[0]]
values=[evaluations[r]['calibrated']['by_slice']['in_distribution']['joint_macro_accuracy']*100 for r in names]
axes[1].bar(labels,values,color=['#becbd7',BLUE,TEAL],width=.65)
for i,value in enumerate(values):axes[1].text(i,value+3,f'{value:.1f}%',ha='center',weight='bold',fontsize=13)
axes[1].set(title='Held out: are both inputs needed?',ylabel='Joint accuracy (%)',ylim=(0,105))
for ax in axes:ax.set_facecolor(BG);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.065,.07,'Matched controls use the same architecture, initialization, re-pairing stream, and optimizer-step count. No real-browser claim.',fontsize=10,color=INK)
save(fig,'results')

fig,ax=plt.subplots(figsize=(11,5.6),facecolor=BG);fig.subplots_adjust(top=.72,bottom=.24,left=.09,right=.97)
fig.text(.07,.91,'Transfer within the controlled world',fontsize=22,weight='bold',color=INK)
fig.text(.07,.83,'Report each shift separately; related observations are not independent test samples',fontsize=11,color=INK)
slices=['in_distribution','compositional','paraphrase','position_intervention']
display=['New speakers\n+ new panels','Held command–\ncolor pairs','New question\nphrasing','Moved\ncontrols']
width=.24;x=np.arange(len(slices))
for offset,name,label,color in [(-1,RUNS[1],'Audio only','#aabac9'),(0,RUNS[2],'Image only',BLUE),(1,RUNS[0],'Joint',TEAL)]:
    values=[evaluations[name]['calibrated']['by_slice'][s]['joint_macro_accuracy']*100 for s in slices]
    ax.bar(x+offset*width,values,width,label=label,color=color)
ax.set_xticks(x,display);ax.set(ylabel='Joint accuracy (%)',ylim=(0,100));ax.set_facecolor(BG)
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=3,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.07,.07,'The generated layout, eight spoken words, and learned text vocabulary remain limited. These are not open-world tasks.',fontsize=10,color=INK)
save(fig,'transfer')

scenes=read_manifest(MANIFEST);records=expand_scenes([s for s in scenes if s['split']=='test'])
outputs={r:aligned_predictions(records,read_manifest(ROOT/'reports'/r/'predictions.jsonl')) for r in RUNS}
indices=[i for i,r in enumerate(records) if r['slice']=='in_distribution' and r['task'] in JOINT_TASKS]
speakers=sorted({records[i]['speaker'] for i in indices})
groups={s:[i for i in indices if records[i]['speaker']==s] for s in speakers}
draws=np.random.default_rng(901).integers(0,len(speakers),(2000,len(speakers)))
counts=np.array([len(groups[s]) for s in speakers]);gaps={}
for control in RUNS[1:]:
    totals=np.array([sum(outputs[RUNS[0]][i]['correct']-outputs[control][i]['correct'] for i in groups[s]) for s in speakers])
    samples=totals[draws].sum(1)/counts[draws].sum(1)
    low,high=np.quantile(samples,[.025,.975])
    gaps[control]={'mean_percentage_points':float(totals.sum()/counts.sum()*100),
                   'interval_percentage_points':[float(low*100),float(high*100)],'speaker_clusters':len(speakers),
                   'method':'Paired speaker-cluster percentile bootstrap; fixed checkpoints, not training-seed variance'}
summary={'runs':{r:sha256(ROOT/'reports'/r/'evaluation.json') for r in RUNS},'paired_gaps':gaps,
         'joint_model_parameters':training[RUNS[0]]['parameters'],'test_speakers':len({s['speaker'] for s in scenes if s['split']=='test'}),
         'source_manifest_sha256':sha256(MANIFEST),'native_joint_model_trained':True,'real_browser_generalization_established':False}
write_json(OUT/'summary.json',summary)

lines=['# First unified multimodal decision model','',
       'A single neural model now reads recorded audio, image pixels, question text, and candidate descriptions. One shared scoring head returns probabilities over the supplied answers. All encoders and fusion/scoring layers are jointly trained; the acoustic representation was warm-started from our earlier small speech model, with its fixed output head removed.','',
       '![Measured joint results](results.svg)','',
       '| Held-out slice | Joint model | Audio-only control | Image-only control | Joint questions |',
       '|---|---:|---:|---:|---:|']
for s,label in zip(slices,display):
    a=evaluations[RUNS[0]]['calibrated']['by_slice'][s];b=evaluations[RUNS[1]]['calibrated']['by_slice'][s];c=evaluations[RUNS[2]]['calibrated']['by_slice'][s]
    lines.append(f"| {label.replace(chr(10),' ')} | {a['joint_macro_accuracy']:.2%} | {b['joint_macro_accuracy']:.2%} | {c['joint_macro_accuracy']:.2%} | {a['joint_examples']} |")
lines+=['','The score averages six question families: color/position of the spoken command or its opposite, and presence/absence. Audio-word and tile-word auxiliary tasks are excluded. Evaluation speakers were never used in any earlier baseline partition.','',
        '**The held-composition test fails:** accuracy falls to 45.66%, below the unimodal controls. The model uses both inputs on familiar combinations but does not yet separate command identity from color robustly. Further work must treat this exposed slice as development evidence and use a fresh confirmation set.','',
        '![Transfer slices](transfer.svg)','',
        '## What the comparison establishes','']
for name,gap in gaps.items():
    lo,hi=gap['interval_percentage_points']
    lines.append(f"- Joint minus {name}: **{gap['mean_percentage_points']:.1f} percentage points**, paired speaker-cluster interval **[{lo:.1f}, {hi:.1f}]** on in-distribution held-out questions.")
lines+=['','These intervals condition on one set of trained checkpoints. They do not estimate variability across training seeds, and they do not establish general browser capability. Related question and intervention variants share observations; counts must not be read as independent recordings.','',
        '## Candidate and question behavior','',
        f"Candidate permutation changed logits by at most **{full['candidate_permutation_max_logit_difference']:.3g}**. Answering the first question alone versus in the batch changed logits by at most **{full['question_batch_isolation_max_logit_difference']:.3g}**. IDs are ignored by the neural encoder. Candidate-subset checks use 2–4 options without retraining; they are easier-choice structural diagnostics, not tests of unseen vocabulary.",'',
        '| Same observations, different questions | Pairs requiring different answers | Both answered correctly |',
        '|---|---:|---:|']
for pair,result in full['question_contrasts'].items():lines.append(f"| {pair} | {result['pairs_with_different_answers']} | {result['both_correct_rate']:.2%} |")
lines+=['','## Why the first attempt was retained','',
        'Fixed one-to-one recording/panel pairs overfit. The second attempt changed only training pairing: draw a word and one of its recordings independently of the panel, then recompute the supervised answer. This breaks both recording→panel and panel→spoken-word memorization. Architecture and optimizer stayed fixed; the first development run remains in the repository. Selection and nomination used development data before final metrics were opened.','',
        '## Performance scope','',
        f"The model has **{training[RUNS[0]]['parameters']:,} parameters**. Its measured loaded-model pipeline p95 is **{full['timing']['warm_pipeline_p95_ms']:.2f} ms** across 32 examples, including file decoding, preprocessing, question/candidate encoding, fusion, scoring, and device synchronization. Listening duration and checkpoint loading are excluded. This is offline one-second speech, not streaming latency.",'',
        'The visuals are generated 2×2 symbol panels, not real application screenshots. Speech is eight isolated keywords. The learned text vocabulary has 43 tokens including special tokens. All candidate concepts and question families occur in training; held-out compositions and wording test limited recombination. This independently designed supervised model does not reproduce Jev’s unpublished RLCD or establish arbitrary instruction following.','',
        'Temperature scaling was fitted on separate calibration data. Raw and adjusted NLL, Brier, reliability bins, and per-task accuracy are in the full evaluation files; an accuracy result alone does not establish calibration under shift.','',
        '## Evidence and reproduction','',
        '- [Architecture and interface](../../docs/research/native-joint-model.md)',
        '- [Initial protocol](../../evals/joint-protocol-v1.json) and [re-pairing follow-up](../../evals/joint-protocol-v2.json)',
        '- [Pre-test nomination](../../evals/joint-nomination-v2.json)',
        '- [Full evaluation](../joint-full-v2/evaluation.json) and [complete predictions](../joint-full-v2/predictions.jsonl)',
        '- [Audio-only evaluation](../joint-audio-only-v2/evaluation.json)',
        '- [Image-only evaluation](../joint-image-only-v2/evaluation.json)',
        '- [First attempt, development only](../joint-full-v1/training.json)',
        '- [Checkpoint](../../artifacts/joint-full-v2/model.safetensors) and [config](../../artifacts/joint-full-v2/config.json)','',
        '```bash','uv run --locked mmso prepare speech_keywords','uv run --locked mmso joint-prepare','uv run --locked python scripts/run_joint_demo.py','uv run --locked python scripts/check_joint_results.py','```','',
        'The demo uses the first in-distribution test scene, chosen before held-out evaluation. Its five questions are supplied through the public prediction function. The symbolic oracle is not called by that function.','']
(OUT/'README.md').write_text('\n'.join(lines))
print('Rendered measured joint-model report and plots')
