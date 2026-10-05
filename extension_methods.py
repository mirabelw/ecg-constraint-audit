"""Signed tangent variants and transparent, dose-matched random comparator."""
import numpy as np,torch
from geometry import P,project
METHODS=['saliency','input_gradient','integrated_gradients','smoothgrad','tangent_saliency','tangent_input_gradient','tangent_integrated_gradients']

def maps(model,x,task,ids,steps=32,smooth_samples=8):
    t=torch.from_numpy(x).requires_grad_(True);g=torch.autograd.grad(model(t)[:,task].sum(),t)[0].detach().numpy();pg=project(g)
    integral=np.zeros_like(x)
    for i in range(steps):
        u=(t.detach()*((i+.5)/steps)).requires_grad_(True);integral+=torch.autograd.grad(model(u)[:,task].sum(),u)[0].detach().numpy()/steps
    sm=np.zeros_like(x)
    # Noise realizations per patient, independent of batch size or model seed.
    noises=np.stack([np.random.default_rng(914+int(ecg)).normal(0,.1,size=(smooth_samples,12,1000)).astype('float32') for ecg in ids])
    for i in range(smooth_samples):
        u=torch.from_numpy(x+project(noises[:,i]).astype('float32')).requires_grad_(True);sm+=torch.autograd.grad(model(u)[:,task].sum(),u)[0].detach().numpy()/smooth_samples
    arrays=[np.abs(g),np.abs(x*g),np.abs(x*integral),np.abs(sm),np.abs(pg),np.abs(x*pg),np.abs(x*project(integral))]
    return np.stack([a.reshape(len(x),12,20,50).mean(-1) for a in arrays],axis=1)

def amplitude_random_mask(delta,guide_mask,ids,method_index):
    """Random window matching of log squared amplitude, without replacement.

    Returns a mask, pre-rescale norm ratio and overlap. Delta is the full
    replacement signal (e.g. -x or mean(window)-x), not the selected delta.
    A caller scales the selected comparator to the guide's exact L2 norm.
    """
    energy=(delta.reshape(len(delta),12,20,50)**2).sum(-1)[:,:6].reshape(len(delta),-1);guide=guide_mask[:,:,::50][:,:6].reshape(len(delta),-1)>0
    out=np.zeros((len(delta),12,20),dtype='float32');ratios=[];overlap=[]
    for b in range(len(delta)):
        rng=np.random.default_rng(1000000+int(ids[b])+method_index*100003);selected=np.flatnonzero(guide[b]);selected=selected[np.argsort(-energy[b,selected],kind='stable')];available=np.ones(120,dtype=bool);chosen=[]
        for target in selected:
            pool=np.flatnonzero(available);dist=np.abs(np.log(energy[b,pool]+1e-12)-np.log(energy[b,target]+1e-12));candidates=pool[np.argsort(dist,kind='stable')[:5]];pick=int(rng.choice(candidates));available[pick]=False;chosen.append(pick)
        chosen=np.array(chosen);out[b,chosen//20,chosen%20]=1;ratios.append(np.sqrt(energy[b,chosen].sum()/max(energy[b,selected].sum(),1e-12)));overlap.append(np.isin(chosen,selected).mean())
    return np.repeat(out,50,-1),np.array(ratios),np.array(overlap)

def dose_operators(full_delta,mask,dose=None):
    raw=(full_delta*mask).astype('float32');norm=lambda d:np.sqrt((d*d).sum((1,2)))
    natural=norm(raw)
    if dose is None:dose=natural
    raw*=np.divide(dose,natural,out=np.zeros_like(natural),where=natural>1e-12)[:,None,None]
    projected=project(raw).astype('float32');pnorm=norm(projected)
    energy=projected*np.divide(dose,pnorm,out=np.zeros_like(dose),where=pnorm>1e-12)[:,None,None]
    normal_fraction=((raw-projected)**2).sum((1,2))/np.maximum((raw*raw).sum((1,2)),1e-20)
    return raw,energy,projected,normal_fraction
