"""Known acquisition algebra. All operators act in common physical units."""
import numpy as np
A=np.array([[1,0],[0,1],[-1,1],[-.5,-.5],[1,-.5],[-.5,1]],dtype=np.float64)
P=A@np.linalg.inv(A.T@A)@A.T
TARGET=np.zeros((64,6,6))
for bits in range(64):
    s=np.flatnonzero([(bits>>j)&1 for j in range(6)])
    if len(s):
        # Minimum Euclidean norm correction, satisfying delta[S] = -x[S].
        TARGET[bits][:,s]=-P[:,s]@np.linalg.pinv(P[np.ix_(s,s)],rcond=1e-10)

def project(x):
    y=x.copy();y[...,:6,:]=np.einsum('ij,...jt->...it',P,x[...,:6,:]);return y

def interventions(x, mask):
    """x and mask: [batch,lead,time]. Returns raw, target-, energy-matched deltas.

    Target matching is a COUPLED intervention, not isolated lead importance.
    Energy matching preserves the direction closest to the raw intervention,
    rescaled to its total L2 dose; selected coordinates are not exactly erased.
    """
    raw=-x*mask
    projected=project(raw)
    raw_norm=np.linalg.norm(raw.reshape(len(x),-1),axis=1)
    projected_norm=np.linalg.norm(projected.reshape(len(x),-1),axis=1)
    ratio=np.divide(raw_norm,projected_norm,out=np.zeros_like(raw_norm),where=projected_norm>1e-12)
    energy=projected*ratio[:,None,None]
    bits=np.sum(mask[:,:6,:].astype(np.int64)*(2**np.arange(6))[None,:,None],axis=1)
    target=raw.copy()
    target[:,:6,:]=np.einsum('btij,bjt->bit',TARGET[bits],x[:,:6,:])
    return {'independent_zero':raw,'target_matched':target,'energy_matched':energy}

def diagnostics(x,mask,delta):
    """Dose ratios, collateral energy fraction, intended-target relative error."""
    raw=-x*mask
    norm=lambda a:np.linalg.norm(a.reshape(len(a),-1),axis=1)
    denom=np.maximum(norm(raw),1e-12)
    total=np.maximum(norm(delta)**2,1e-24)
    return {'dose_ratio':norm(delta)/denom,
            'collateral_fraction':norm(delta*(1-mask))**2/total,
            'selected_relative_error':norm((delta-raw)*mask)/denom,
            'off_subspace_rms_mv':np.sqrt(np.mean((delta-project(delta))**2,axis=(1,2)))}

def validate():
    rng=np.random.default_rng(13);x=project(rng.normal(size=(64,12,30)))
    mask=np.zeros_like(x)
    for bits in range(64):
        mask[bits,:6,:]=np.array([(bits>>j)&1 for j in range(6)])[:,None]
        mask[bits,6+(bits%6),3:8]=1
    d=interventions(x,mask)
    assert np.max(np.abs((x+d['target_matched'])*mask))<1e-10
    for name in ['target_matched','energy_matched']:
        assert np.max(np.abs(d[name]-project(d[name])))<1e-10
    assert np.allclose(np.linalg.norm(d['energy_matched'].reshape(64,-1),axis=1),np.linalg.norm(d['independent_zero'].reshape(64,-1),axis=1))
    chest=mask.copy();chest[:,:6]=0;dc=interventions(x,chest)
    assert np.allclose(dc['independent_zero'],dc['target_matched'])
    assert np.allclose(dc['independent_zero'],dc['energy_matched'])
    # Positive control: model depending only on impossible residual has zero
    # response on consistent directions, nonzero response on raw limb deletion.
    v=np.array([1,-1,1,0,0,0.]);assert np.allclose(v@P,0)
    residual=lambda y:np.einsum('l,blt->bt',v,y[:,:6]).sum(1)
    raw_response=np.abs(residual(x+d['independent_zero'])-residual(x))
    assert raw_response.max()>1
    assert np.abs(residual(x+d['energy_matched'])-residual(x)).max()<1e-10
    return {'all_64_limb_masks':'pass','selected_zero':'pass','energy_dose':'pass','chest_identity':'pass','residual_only_positive_control':'pass'}

if __name__=='__main__':print(validate())
