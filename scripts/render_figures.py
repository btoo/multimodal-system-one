"""Original documentation figures. No model training or measured model results.

Run: uv run --locked --group docs python scripts/render_figures.py
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#f8fafc"
INK = "#14283e"
MUTED = "#506477"
TEAL = "#007e78"
BLUE = "#2e62bc"
GOLD = "#b26614"
PURPLE = "#7850a8"
RED = "#ba4563"
PALE = {TEAL: "#dff2ed", BLUE: "#e6eefc", GOLD: "#fff0d9", PURPLE: "#f0e9f7", RED: "#fbe7ec"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "svg.fonttype": "none", "svg.hashsalt": "mm-system-one-v1",
    "axes.titleweight": "bold", "figure.facecolor": BG, "axes.facecolor": BG})
MANIFEST = {}


def canvas(kicker, title, subtitle="", height=8):
    fig = plt.figure(figsize=(14, height), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 14), ylim=(0, height))
    ax.set_axis_off()
    ax.text(.55, height-.40, kicker.upper(), fontsize=10, color=TEAL, weight="bold", va="top")
    ax.text(.55, height-.83, title, fontsize=23, weight="bold", va="top")
    if subtitle:
        ax.text(.55, height-1.36, subtitle, fontsize=11, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=TEAL, dashed=False, title_size=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=.12",
        lw=1.25, ec=color, fc=PALE[color], linestyle="--" if dashed else "-"))
    ax.text(x+.18, y+h-.22, title, fontsize=title_size, color=INK, weight="bold", va="top")
    if body:
        ax.text(x+.18, y+h-.62, body, fontsize=11, color=MUTED, va="top", linespacing=1.45)


def arrow(ax, start, end, color=MUTED, rad=0, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
        lw=1.6, color=color, connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-", shrinkA=5, shrinkB=5))


def note(ax, text):
    ax.text(.55, .27, text, fontsize=10, color=MUTED, va="center")


def save(fig, name, kind, description):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=160, facecolor=BG,
                    metadata={"Date": None} if ext == "svg" else None)
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)
    MANIFEST[name] = {"kind": kind, "description": description}


def overview():
    f, a = canvas("01 / research target", "Perception → probabilities → decisions",
                  "Application target: real speech, acoustic events, and screen understanding")
    box(a, .6, 4.2, 3.25, 1.15, "Visual observation", "Pixels → spatial patch tokens", BLUE)
    box(a, .6, 2.7, 3.25, 1.15, "Question + candidates", "Language → semantic tokens", PURPLE)
    box(a, .6, 1.25, 3.25, 1.0, "Speech + sounds", "Acoustic features + timestamps", GOLD)
    box(a, 5.05, 2.7, 3.8, 2.65, "Shared decision network",
        "Native input representations\nTrainable fusion and heads\nSeparate from-scratch controls", TEAL)
    box(a, 10.05, 4.15, 3.3, 1.2, "Answer probabilities", "Choice · boolean\nOrdered levels · event labels", BLUE)
    box(a, 10.05, 2.4, 3.3, 1.2, "Application policy", "Choose an action or abstain", GOLD)
    arrow(a, (3.85, 4.8), (5.05, 4.4)); arrow(a, (3.85, 3.3), (5.05, 3.6))
    arrow(a, (3.85, 1.75), (5.05, 2.9))
    arrow(a, (8.85, 4.65), (10.05, 4.75)); arrow(a, (11.7, 4.15), (11.7, 3.6))
    a.text(5.12, 1.5, "Training signal", fontsize=12, weight="bold", color=TEAL)
    a.text(5.12, 1.05, "Known outcomes → proper probability loss", fontsize=12)
    arrow(a, (6.9, 1.65), (6.9, 2.7), TEAL)
    note(a, "A probability model and a decision policy are separate objects. No trained-model result is shown.")
    save(f, "overview", "design", "Proposed multimodal prediction and policy separation.")


def fusion():
    f, a = canvas("02 / architecture search", "Four credible ways to combine modalities",
                  "Image/text synthetic control · four inductive biases to study before extending the comparison")
    cols = [(.55, "A0 · Late fusion", BLUE), (3.92, "A1 · FiLM", GOLD),
            (7.29, "A2 · Joint tokens", TEAL), (10.66, "A3 · Latent array", PURPLE)]
    bodies = ["Pool each modality,\nthen combine vectors.", "Text scales and shifts\nvisual feature maps.",
              "Image and text tokens\ninteract at every layer.", "Inputs write into a\nsmall shared latent set."]
    for i, (x, title, c) in enumerate(cols):
        box(a, x, 1.15, 2.78, 4.9, title, color=c, title_size=14)
        for j, label in enumerate(["image", "text"]):
            a.add_patch(Rectangle((x+.23+j*1.17, 4.65), 1.0, .48, fc=PALE[BLUE if j==0 else PURPLE], ec=MUTED, lw=.7))
            a.text(x+.73+j*1.17, 4.89, label, ha="center", va="center", fontsize=11)
        if i == 0:
            for j in range(2):
                arrow(a, (x+.7+j*1.18, 4.65), (x+.7+j*1.18, 3.7))
                a.add_patch(Circle((x+.7+j*1.18, 3.45), .22, color=c))
            arrow(a, (x+.8, 3.15), (x+1.3, 2.8)); arrow(a, (x+1.9, 3.15), (x+1.5, 2.8))
        elif i == 1:
            a.text(x+1.39, 3.55, "γ(text) × vision\n+ β(text)", ha="center", fontsize=13, color=c)
            arrow(a, (x+.7, 4.65), (x+.7, 4.1)); arrow(a, (x+1.9, 4.65), (x+1.9, 4.1))
        elif i == 2:
            for row in range(3):
                for col in range(6):
                    a.add_patch(Rectangle((x+.25+col*.38, 3.05+row*.38), .28, .26,
                        fc=BLUE if (col+row)%2==0 else PURPLE, alpha=.78))
            arrow(a, (x+1.4, 4.65), (x+1.4, 4.05))
        else:
            for j in range(4):
                a.add_patch(Circle((x+.53+j*.56, 3.5), .16, color=c))
            arrow(a, (x+.7, 4.65), (x+1.0, 3.8)); arrow(a, (x+1.9, 4.65), (x+1.65, 3.8))
            a.text(x+1.4, 2.95, "M ≪ input length", ha="center", fontsize=12, color=c)
        a.text(x+.19, 2.25, bodies[i], fontsize=11, color=MUTED, va="top", linespacing=1.5)
    note(a, "A2 is the proposed implementation baseline. Architecture superiority is still an experimental question.")
    save(f, "fusion-families", "design", "Four architecture families to compare under the same contract.")


def architecture():
    f, a = canvas("03 / proposed A2 synthetic control", "A direct, question-conditioned candidate scorer",
                  "64×64 is a toy workload · real screens and speech have a separate evaluation contract")
    box(a, .65, 4.9, 3.6, 1.13, "64 × 64 RGB", "8 × 8 patches → 64 visual tokens", BLUE)
    box(a, 5.2, 4.9, 3.6, 1.13, "Question text", "Token + position embeddings", PURPLE)
    box(a, 4.0, 2.7, 5.0, 1.4, "Joint transformer → H",
        "Image + question + positions\nProposed: width 256 · 6 blocks", TEAL)
    arrow(a, (2.5, 4.9), (4.9, 4.1)); arrow(a, (7.0, 4.9), (7.0, 4.1))
    box(a, .65, .95, 3.0, 1.15, "Candidate descriptions", "Same text encoder for each", PURPLE)
    box(a, 4.6, .95, 4.4, 1.15, "Read H → shared score g", "One scalar per candidate", TEAL)
    box(a, 10.05, .95, 3.25, 1.15, "Mask → softmax", "Probability distribution", BLUE)
    arrow(a, (3.65, 1.5), (4.6, 1.5)); arrow(a, (6.9, 2.7), (6.9, 2.1)); arrow(a, (9, 1.5), (10.05, 1.5))
    box(a, 10.05, 3.05, 3.25, 2.5, "Structural properties",
        "Candidate order equivariance\nQuestion isolation by batching\nJoint gradients to input stems\nCalibration fitted afterward", GOLD, title_size=13)
    note(a, "Model size is a proposal (roughly 5–10M parameters), not an implemented count or measured performance.")
    save(f, "architecture", "design", "Proposed joint transformer and independent semantic candidate scoring.")


def candidate_equivariance():
    f, a = canvas("04 / a testable invariant", "Reordering answers must only reorder probabilities",
                  "Illustrative logits [3, 2, 1] · deterministic inference · candidate IDs carry no evidence")
    names = ["red", "blue", "green"]
    colors = [RED, BLUE, TEAL]
    probs = np.exp([3., 2., 1.]); probs /= probs.sum()
    for x, order, title in [(1, [0, 1, 2], "Original candidate order"), (8, [2, 0, 1], "Permuted candidate order")]:
        a.text(x, 5.75, title, weight="bold", fontsize=15)
        for row, k in enumerate(order):
            y = 4.6-row*1.1
            a.text(x, y+.18, names[k], fontsize=13)
            a.add_patch(Rectangle((x+1.0, y), probs[k]*4, .52, color=colors[k]))
            a.text(x+1.08+probs[k]*4, y+.26, f"{probs[k]:.3f}", fontsize=12, va="center")
    arrow(a, (6.0, 3.85), (7.25, 3.85), TEAL)
    a.text(7, 1.3, r"$p(x,q,\pi C)=\pi p(x,q,C)$", ha="center", fontsize=23, color=TEAL)
    note(a, "The same candidate set is assumed. Adding candidates changes normalization; ties need order-independent handling.")
    save(f, "candidate-equivariance", "analytic", "Softmax values of fixed illustrative logits; not model predictions.")


def chart_base(kicker, title, subtitle, count=1):
    f, a = canvas(kicker, title, subtitle)
    axes = [f.add_axes([.075, .20, .88, .53])] if count==1 else [f.add_axes([.075,.20,.39,.53]), f.add_axes([.565,.20,.39,.53])]
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        for sp in ['left','bottom']:ax.spines[sp].set_color('#c5ced8')
        ax.grid(alpha=.2, color=MUTED)
        ax.set_axisbelow(True)
    return f, a, axes


def scaling():
    f, a, (ax,) = chart_base("05 / analytical cost model", "Long inputs make compression worth testing",
                             "Attention-pair counts only · 6 processing layers · 32 latents · 8 output queries")
    n = np.geomspace(32, 4096, 180)
    full = 6*n*n
    latent = n*32+6*32**2+8*32
    ax.loglog(n, full, color=BLUE, lw=3, label=r"Full attention: $6N^2$")
    ax.loglog(n, latent, color=TEAL, lw=3, label=r"Latent route: $32N+6(32^2)+8(32)$")
    ax.set(xlabel="Input tokens N", ylabel="Attention interactions (arbitrary count)")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    ax.set_xticks([32,64,128,256,512,1024,2048,4096],labels=[32,64,128,256,512,1024,2048,4096])
    note(a, "Not measured speed. Projection/MLP cost, head count, width, data movement, and compression errors are excluded.")
    save(f, "attention-scaling", "analytic", "Pair counts: 6*N^2 versus N*32+6*32^2+8*32. Not a hardware benchmark.")


def proper_scoring():
    f, a, (left, right) = chart_base("06 / objective design", "An honest belief and a useful action can differ",
                        "Analytical example: the event happens with true probability q = 0.70", 2)
    p = np.linspace(.005,.995,700); q=.7
    nll = -q*np.log(p)-(1-q)*np.log(1-p)
    best_nll = -q*np.log(q)-(1-q)*np.log(1-q)
    brier_regret=(p-q)**2
    left.plot(p,nll-best_nll,color=BLUE,lw=2.5,label="Excess expected log loss")
    left.plot(p,brier_regret,color=TEAL,lw=2.5,label="Excess expected Brier loss")
    left.axvline(q,color=MUTED,ls="--",alpha=.7)
    left.scatter([q],[0],color=TEAL,zorder=4,s=65)
    left.set(xlabel="Reported probability p",ylabel="Excess loss above optimum",ylim=(-.035,1.15),xlim=(0,1))
    left.legend(frameon=False,fontsize=10,loc="upper center")
    left.set_title("Probability estimation",fontsize=14,pad=18)
    x=np.linspace(0,1,101)
    reward=q*x+(1-q)*(1-x)
    right.plot(x,reward,color=GOLD,lw=3)
    right.scatter([1],[q],color=GOLD,s=65,zorder=4)
    right.axvline(q,color=MUTED,ls="--",alpha=.7)
    right.set(xlabel="Probability of choosing 'yes'",ylabel="Expected correctness reward",xlim=(0,1.03),ylim=(.25,.75))
    right.set_title("Action selection",fontsize=14,pad=18)
    right.text(.08,.67,"Best action: always say yes\nHonest belief: 70%",fontsize=12)
    note(a, "Closed-form objectives, not training results. This calculation does not describe Jev's unpublished RLCD implementation.")
    save(f,"proper-scoring","analytic","True q=.7; log/Brier regret is minimized at p=q; sampled correctness reward is maximized at p=1.")


def calibration():
    rng=np.random.default_rng(20260923)
    q=rng.beta(2,2,40000)
    y=rng.binomial(1,q)
    logits=np.log(q/(1-q))
    raw=1/(1+np.exp(-2*logits))
    corrected=q
    f,a,(left,right)=chart_base("07 / synthetic diagnostics", "Confidence needs its own evaluation",
              "40,000 simulated Bernoulli outcomes · artificial sharpening · known inverse temperature",2)
    bins=np.linspace(0,1,11)
    counts=[]
    for pred,label,c in [(raw,"Artificially overconfident",RED),(corrected,"Known-temperature correction",TEAL)]:
        xs=[];ys=[];ns=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(pred>=lo)&(pred<hi)
            xs.append(pred[m].mean());ys.append(y[m].mean());ns.append(int(m.sum()))
        left.plot(xs,ys,"o-",color=c,lw=2,label=label)
        counts.append(ns)
    left.plot([0,1],[0,1],"--",color=MUTED,lw=1,label="Ideal reliability")
    left.set(xlabel="Mean predicted event probability",ylabel="Observed event frequency",xlim=(0,1),ylim=(0,1))
    left.legend(frameon=False,fontsize=9,loc="upper left")
    conf=np.maximum(q,1-q);errors=((q>=.5)!=y).astype(float)
    order=np.argsort(-conf);cumulative=np.cumsum(errors[order])
    sizes=np.linspace(500,len(y),100).astype(int)
    risk=cumulative[sizes-1]/sizes
    right.plot(sizes/len(y),risk,color=BLUE,lw=3)
    right.set(xlabel="Coverage: fraction accepted",ylabel="Error rate among accepted",xlim=(0,1),ylim=(0,.36))
    right.text(.1,.31,"Reject more cases ←\n→ Accept more cases",fontsize=11)
    note(a,"Simulated illustration only. This oracle correction is not evidence that post-hoc calibration fixes a learned model.")
    save(f,"calibration","simulation","Seed 20260923; q~Beta(2,2); y~Bernoulli(q); raw=sigmoid(2*logit(q)); corrected=q.")
    MANIFEST['calibration']['bin_counts']={"raw":counts[0],"corrected":counts[1]}
    MANIFEST['calibration']['samples']=40000
    MANIFEST['calibration']['seed']=20260923


def scene(ax,x,y,swap=False):
    ax.add_patch(FancyBboxPatch((x,y),3.0,1.8,boxstyle="round,pad=.04",fc="white",ec="#cad5df"))
    redx=x+(2.2 if swap else .65)
    bluex=x+(.45 if swap else 2.0)
    ax.add_patch(Circle((redx,y+.93),.26,fc=RED))
    ax.add_patch(Rectangle((bluex,y+.65),.52,.52,fc=BLUE))
    ax.add_patch(Circle((x+1.4,y+.38),.16,fc=TEAL))


def counterfactuals():
    f,a=canvas("08 / causal task controls", "Change the evidence. Check what must change.",
               "Three hand-constructed illustrations of a relation task · oracle answers, not predictions")
    rows=[(.55,"Base scene",False,"Is the red circle LEFT\nof the blue square?","TRUE"),
          (5.0,"Change only the image",True,"Is the red circle LEFT\nof the blue square?","FALSE"),
          (9.45,"Change only the question",False,"Is the red circle RIGHT\nof the blue square?","FALSE")]
    for x,title,swap,q,answer in rows:
        a.text(x,5.85,title,fontsize=14,weight="bold")
        scene(a,x+.3,3.45,swap)
        a.text(x+.2,2.85,q,fontsize=12,va="top",linespacing=1.5)
        a.text(x+.2,1.65,answer,fontsize=20,weight="bold",color=TEAL if answer=="TRUE" else RED)
    note(a,"Also change irrelevant distractors: predictions should then stay stable. Split related scene variants together.")
    save(f,"counterfactuals","design","Hand-built task illustrations with deterministic scene-oracle labels.")


def research_loop():
    f,a=canvas("09 / experiment methodology", "Keep the evidence—and several routes forward",
               "Proposed archive-based search · each candidate is tied to a hypothesis, source revision, and budget")
    box(a,.55,4.4,3.0,1.3,"Evidence + archive","Sources · successes · failures",PURPLE)
    box(a,5.0,4.4,3.1,1.3,"Testable hypothesis","One change + expected effect",BLUE)
    box(a,9.55,4.4,3.7,1.3,"Controlled trial","Fixed data and resource contract",TEAL)
    arrow(a,(3.55,5.05),(5.0,5.05));arrow(a,(8.1,5.05),(9.55,5.05))
    box(a,9.55,1.6,3.7,1.55,"Independent measurement","Quality · latency · memory\nUncertainty · counterexamples",BLUE)
    box(a,5.0,1.6,3.1,1.55,"Analysis + confirmation","What changed? What failed?\nRepeat with fresh seeds",GOLD)
    box(a,.55,1.6,3.0,1.55,"Archive update","Keep complementary branches\nRetain failed-trial evidence",PURPLE)
    arrow(a,(11.4,4.4),(11.4,3.15));arrow(a,(9.55,2.4),(8.1,2.4));arrow(a,(5,2.4),(3.55,2.4))
    arrow(a,(2,3.15),(2,4.4),PURPLE)
    note(a,"Stop at the campaign cap. A lower development score is provisional until confirmed and independently evaluated.")
    save(f,"research-loop","design","Proposed evidence-driven population search; no loop has executed.")


def firewall():
    f,a=canvas("10 / evaluation design", "Four data roles. Four different permissions.",
               "Scene families are separated before rendering, paraphrasing, or candidate generation")
    entries=[(.6,"TRAIN",TEAL,"Update model weights\nGenerate learning examples"),
             (3.95,"DEVELOPMENT",BLUE,"Compare hypotheses\nChoose a model"),
             (7.3,"CALIBRATION",GOLD,"Fit temperature\nSet decision thresholds"),
             (10.65,"FINAL TEST",PURPLE,"Report frozen decisions\nNo tuning feedback")]
    for x,title,c,b in entries:box(a,x,3.2,2.72,2.65,title,b,c,title_size=13)
    for x in (3.32,6.67,10.02):arrow(a,(x,4.5),(x+.62,4.5))
    a.text(.65,2.25,"Optimize",color=TEAL,weight="bold",fontsize=13)
    a.text(4.0,2.25,"Select",color=BLUE,weight="bold",fontsize=13)
    a.text(7.35,2.25,"Adjust probabilities",color=GOLD,weight="bold",fontsize=13)
    a.text(10.7,2.25,"Measure transfer",color=PURPLE,weight="bold",fontsize=13)
    a.text(.65,1.2,"A final test used to revise the design becomes development evidence for the next campaign.",fontsize=13)
    note(a,"Procedural separation only. A file named 'test' is not a security boundary, and repeated selection can still overfit.")
    save(f,"evaluation-firewall","design","Four distinct split roles with a one-way model-selection flow.")


def two_level():
    f,a=canvas("11 / research about research", "A better model is not yet a better researcher",
               "Two separately versioned objects · two evaluation suites · resource accounting at both levels")
    box(a,.65,4.0,3.1,1.55,"Research policy φ","Proposal · selection · analysis\nLater: editable method",PURPLE)
    box(a,5.0,4.0,3.6,1.55,"New task-suite trials","Compare policy versions\nMatched total budgets",PURPLE)
    box(a,9.85,4.0,3.5,1.55,"Method transfer check","More discovery per budget?\nPromote only with evidence",GOLD)
    arrow(a,(3.75,4.8),(5,4.8));arrow(a,(8.6,4.8),(9.85,4.8))
    box(a,.65,1.2,3.1,1.55,"Candidate model θ","Architecture · loss · training\nFirst implementation target",TEAL)
    box(a,5.0,1.2,3.6,1.55,"Model experiments","Train on controlled scenes\nScore on fixed development",TEAL)
    box(a,9.85,1.2,3.5,1.55,"Model transfer check","New combinations and seeds\nFinal quality and calibration",BLUE)
    arrow(a,(3.75,2),(5,2));arrow(a,(8.6,2),(9.85,2))
    arrow(a,(2.15,4),(2.15,2.75),PURPLE)
    a.text(2.45,3.28,"guides",fontsize=11,color=PURPLE)
    note(a,"The first campaign holds φ fixed. Its optional evolution needs a separate budget and a held-out family of tasks.")
    save(f,"two-level-research","design","Proposed separation of model optimization and later research-policy optimization.")


def eval_suites():
    f,a=canvas("12 / application evaluation", "Four capabilities. Separate evidence for each.",
               "All suites planned · real data and paired cases must be evaluated before claiming application readiness")
    box(a,.6,4.2,3.7,1.6,"Speech meaning","Intent · arguments · corrections\nNew speakers and noise",GOLD)
    box(a,5.1,4.2,3.7,1.6,"Acoustic events","Alerts · speech · music · overlap\nSilence and false activations",PURPLE)
    box(a,9.6,4.2,3.7,1.6,"Screen understanding","State · targets · small UI text\nNew apps and layouts",BLUE)
    box(a,3.4,1.7,7.1,1.5,"Joint audio + screen decisions",
        "Correct action + target · context dependence · ambiguity · abstention",TEAL)
    arrow(a,(2.45,4.2),(4.5,3.2),GOLD)
    arrow(a,(6.95,4.2),(6.95,3.2),PURPLE)
    arrow(a,(11.45,4.2),(9.4,3.2),BLUE)
    a.text(7,.83,"Later: execute the workflow and verify the outcome",ha="center",fontsize=14,weight="bold")
    note(a,"Speech recognition, sound classification, visual grounding, and completed computer tasks are distinct measurements.")
    save(f,"eval-suites","design","Planned real-data capability suites; no measured evaluation results.")


def audio_screen_architecture():
    f,a=canvas("13 / practical native-input model", "Listen to the command. Read the screen. Decide.",
               "Compact pretrained input encoders are a starting hypothesis; fusion and decision heads are trained")
    box(a,.55,4.7,3.4,1.2,"Screenshot","Global view + legible tiles",BLUE)
    box(a,.55,2.95,3.4,1.2,"Speech / audio","Waveform-derived features",GOLD)
    box(a,.55,1.2,3.4,1.2,"Question / history","Context + candidate semantics",PURPLE)
    box(a,4.8,4.7,3.15,1.2,"Visual encoder","Preserve spatial positions",BLUE)
    box(a,4.8,2.95,3.15,1.2,"Acoustic encoder","Preserve time + presence",GOLD)
    box(a,9.0,2.95,4.3,2.95,"Trainable fusion + heads",
        "Speech intent\nIndependent sound-event labels\nScreen state / target scores\nAction + target or abstention",TEAL)
    arrow(a,(3.95,5.3),(4.8,5.3));arrow(a,(3.95,3.55),(4.8,3.55))
    arrow(a,(7.95,5.3),(9.0,5.0),BLUE);arrow(a,(7.95,3.55),(9.0,3.7),GOLD)
    arrow(a,(3.95,1.8),(9.0,3.1),PURPLE)
    a.text(9.1,1.8,"Compare with ASR + vision",fontsize=13,weight="bold")
    a.text(9.1,1.33,"Include encoder and proposal costs",fontsize=11,color=MUTED)
    note(a,"Proposed architecture. Frozen encoders are not jointly learned here; any target proposal stage needs its own recall test.")
    save(f,"audio-screen-architecture","design","Proposed native speech/audio and screenshot fusion, distinct from the synthetic control.")


def implemented_joint():
    f,a=canvas("14 / implemented neural prototype", "One model. A question. Supplied answer meanings.",
               "668,097 trainable parameters · eight recorded keywords · generated panels · learned text vocabulary")
    box(a,.55,4.65,3.2,1.15,"Image pixels","Four fixed quadrants → CNN",BLUE)
    box(a,.55,2.95,3.2,1.15,"Recorded audio","Log-mel → acoustic encoder",GOLD)
    box(a,.55,1.25,3.2,1.15,"Question text","Shared learned text encoder",PURPLE)
    box(a,5.0,2.95,3.35,2.85,"Joint state · 6 tokens",
        "4 visual + position tokens\n1 acoustic representation\n1 question representation\n2 transformer fusion blocks",TEAL)
    arrow(a,(3.75,5.2),(5,5.0),BLUE);arrow(a,(3.75,3.5),(5,3.9),GOLD)
    arrow(a,(3.75,1.8),(5,3.15),PURPLE)
    box(a,9.55,4.65,3.8,1.15,"Candidate descriptions","Same text encoder · arbitrary IDs",PURPLE)
    box(a,9.55,2.45,3.8,1.35,"One shared scalar scorer","Each candidate reads joint state\nSoftmax over supplied answers",TEAL)
    arrow(a,(11.5,4.65),(11.5,3.8),PURPLE);arrow(a,(8.35,4.05),(9.55,3.2),TEAL)
    a.text(5.0,1.75,"Joint end-to-end training",fontsize=14,weight="bold",color=TEAL)
    a.text(5.0,1.2,"Acoustic initialization reused; its fixed class head removed",fontsize=11,color=MUTED)
    note(a,"No transcript, task ID, scene graph, oracle answer, or candidate ID enters the neural forward pass.")
    save(f,"implemented-joint","design","Implemented native candidate scorer. Architecture diagram, not a performance claim.")


def shape_color_factorization():
    f,a=canvas("16 / visual representation experiment", "Test a visual prior separately from model size",
               "An explicit generated-panel assumption: dark glyphs can be separated from the colored tile fill",height=8)
    from mmso.joint_world import render_panel
    colors=["red","green","blue","yellow"]
    panel={"background":245,"tiles":[{"word":"down","color":color,"jitter":[0,0],"stroke":5} for color in colors]}
    pixels=np.array(render_panel(panel))
    a.imshow(pixels,extent=(.7,3.3,2.75,5.35),interpolation="nearest",aspect="auto")
    a.text(2,5.7,"Same glyph, four colors",ha="center",fontsize=13,weight="bold")
    box(a,4.25,4.65,4.2,1.15,"Dark-ink shape map","Fixed pixel transform → learned CNN",BLUE)
    box(a,4.25,2.65,4.2,1.15,"Pooled color","Spatial RGB mean → learned MLP",GOLD)
    box(a,9.25,3.35,4.05,1.75,"One visual token per tile","Concatenate shape + color\nThen use the existing joint model",TEAL,title_size=13)
    arrow(a,(3.3,4.65),(4.25,5.2),BLUE)
    arrow(a,(3.3,3.3),(4.25,3.2),GOLD)
    arrow(a,(8.45,5.2),(9.25,4.65),BLUE)
    arrow(a,(8.45,3.2),(9.25,3.75),GOLD)
    a.text(.7,1.82,"Shape transform:",fontsize=13,weight="bold",color=BLUE)
    a.text(3.0,1.82,"clip((0.4 − max(R, G, B)) / 0.2, 0, 1)",fontsize=13)
    a.text(.7,1.2,"The shape map is invariant to these color interventions. This is a targeted prior, not general object segmentation.",fontsize=11,color=MUTED)
    note(a,"Original generated inputs and architecture diagram. No oracle labels enter inference; performance is reported separately.")
    save(f,"shape-color-prior","design","Renderer-specific fixed dark-ink map and learned shape/color branches; no performance claim.")


def developer_api():
    f,a=canvas("15 / developer interface", "Multimodal content in. Typed decisions out.",
               "HTTP transport wraps the neural scorer; candidate meanings determine the prediction task",height=8.6)
    box(a,.55,4.8,3.3,1.6,"Observation blocks","Image pixels + recorded audio\nQuestion text + answer meanings",BLUE)
    box(a,.55,2.45,3.3,1.65,"Named questions","Choice · Noul · Score · Ranking\nIndependent questions, shared input",PURPLE,title_size=13)
    box(a,4.55,4.8,3.8,1.6,"Validate the model contract","Decode bounded inline media\nCheck vocabulary and input limits",GOLD,title_size=13)
    box(a,4.55,2.45,3.8,1.65,"Cached neural checkpoint","Encode audio and image once\nFuse each question; score candidates",TEAL,title_size=13)
    box(a,9.1,4.8,4.3,1.6,"Typed result per question","Choice → categorical probabilities\nNoul → probability of yes",BLUE)
    box(a,9.1,2.45,4.3,1.65,"Deterministic result views","Score → weighted rubric values\nRanking → sorted candidate scores",PURPLE,title_size=13)
    arrow(a,(3.85,5.6),(4.55,5.6),BLUE)
    arrow(a,(6.45,4.8),(6.45,4.1),GOLD)
    arrow(a,(3.85,3.2),(4.55,3.2),PURPLE)
    arrow(a,(8.35,3.5),(9.1,5.45),TEAL)
    arrow(a,(8.35,3.2),(9.1,3.2),TEAL)
    a.text(.65,1.5,"Model capability is versioned separately from API capability.",fontsize=15,weight="bold",color=TEAL)
    a.text(.65,1.02,"v2: one spoken keyword + generated panel. Structured JSON and low latency do not establish task accuracy.",fontsize=11,color=MUTED)
    note(a,"Interface architecture, not a benchmark. The model predicts distributions; ordinary code applies ranking, rubric values, and abstention.")
    save(f,"developer-api","design","Multimodal input validation, cached neural scorer, and typed deterministic output views.")


if __name__ == "__main__":
    for make in [overview,fusion,architecture,candidate_equivariance,scaling,proper_scoring,
                 calibration,counterfactuals,research_loop,firewall,two_level,eval_suites,audio_screen_architecture,implemented_joint,developer_api,shape_color_factorization]:
        make()
    for name,item in MANIFEST.items():
        item['svg_sha256']=hashlib.sha256((OUT/f'{name}.svg').read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps({"model_results":False,"figures":MANIFEST},indent=2)+'\n')
    print(f"Rendered {len(MANIFEST)} original figures in {OUT}")
