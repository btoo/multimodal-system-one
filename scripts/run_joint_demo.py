"""Run five different questions on one audio/image observation using the saved model."""
import json
from mmso.artifacts import ROOT
from mmso.joint_model import predict_joint

example=json.loads((ROOT/'examples/joint-demo-input.json').read_text())
result=predict_joint(ROOT/example['checkpoint'],ROOT/example['image'],ROOT/example['audio'],example['requests'])
print(json.dumps(result,indent=2))
