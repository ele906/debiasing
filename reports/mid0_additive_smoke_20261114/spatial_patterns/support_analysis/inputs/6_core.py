"""Pattern addition and exact applied-norm matching; no model loading."""
import torch

def apply_pattern(original,z,pattern,coefficient,direction):
    modified=z+coefficient*pattern
    delta=(modified-z)@direction.T
    edited=original+delta.permute(0,3,1,2).to(original)
    return modified,edited,delta


def match_pattern(original,z,pattern,target,direction,tolerance=.005):
    if not torch.isfinite(pattern).all() or float(pattern.norm())==0:
        raise ValueError('Nonfinite or zero pattern cannot reach nonzero target')
    if target<=0:raise ValueError('Target must be positive')
    def evaluate(value):
        _,edited,_=apply_pattern(original,z,pattern,value,direction)
        return float((edited.float()-original.float()).norm())
    lo=0.;hi=2*target/(float(pattern.norm())*float(direction.norm()))
    for _ in range(24):
        if evaluate(hi)>=target:break
        hi*=2
    else:raise ValueError('Could not bracket target norm')
    best=hi;error=abs(evaluate(hi)-target)
    for _ in range(40):
        mid=(lo+hi)/2;norm=evaluate(mid)
        if abs(norm-target)<error:best,error=mid,abs(norm-target)
        if norm<target:lo=mid
        else:hi=mid
    relative=error/target
    return best,relative,relative<=tolerance
