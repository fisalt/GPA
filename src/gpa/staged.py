"""Evaluate the SAME lagged model by level, exposing prefix states by identity reads.

Aligned equal key/slot widths; other shapes/cache segments use the old schedule.
Zero-beta, zero-gate, zero-key/value probes do not update memory. Their identity
queries expose the exact block-boundary states and carry all autograd adjoints.
No manual state detach, token labels, or new storage approximation is used.
"""
import torch
from .ops import gdn_scan,unit,_pad_width
from .state import ResidualSnapshot
from .packed import pack_sequence,read_and_output


def paired_snapshots(model,q,key,slot,value,gates,old):
    b,t,h,d=q.shape;c=model.config.chunk_size;n=t//c
    probe=model._snapshot_basis.shape[0]
    vd=value.shape[-1];pv=max(d,vd)
    qreal=torch.cat((q,torch.zeros_like(q)),2)
    kreal=torch.cat((key,slot),2)
    vreal=torch.cat((_pad_width(slot,pv-d),_pad_width(value,pv-vd)),2)
    gi,bi,go,bo=gates
    greal=torch.cat((gi,go),2);breal=torch.cat((bi,bo),2)
    basis=model._snapshot_basis.to(q.dtype)[None,None,:,None,:].expand(b,n,probe,2*h,d)
    def augment(real,dummy):
        return torch.cat((real.reshape(b,n,c,*real.shape[2:]),dummy),2).reshape(b,n*(c+dummy.shape[2]),*real.shape[2:])
    zero=q.new_zeros(b,n,probe,2*h,d)
    zv=q.new_zeros(b,n,probe,2*h,pv)
    zg=greal.new_zeros(b,n,probe,2*h)
    h0=torch.cat((_pad_width(old.inner,pv-d),_pad_width(old.outer,pv-vd)),1)
    output,final=gdn_scan(augment(qreal,basis),augment(kreal,zero),augment(vreal,zv),
        augment(greal,zg),augment(breal,zg),h0,backend=model.config.backend)
    output=output.reshape(b,n,c+probe,2*h,pv)
    raw=output[:,:,:c,:h,:d].reshape(b,t,h,d)
    snapshots=output[:,:,c:c+d].permute(0,1,3,2,4)
    return raw,final[:,:h,:,:d],snapshots[:,:,:h,:,:d],snapshots[:,:,h:,:,:vd]


def staged_sequence(model,q,k,v,state,control_input=None):
    cfg=model.config;b,t,h,d=q.shape;c=cfg.chunk_size;n=t//c;vd=v.shape[-1]
    frozen=state.frozen or tuple(ResidualSnapshot(s.inner,s.outer) for s in state.levels)
    key,value=k,v
    slots,values,writes,counts,bases,afinals,gos,bos=[],[],[],[],[],[],[],[]
    for j,(old,snap) in enumerate(zip(state.levels,frozen)):
        slot=model._slot(key,j);gates=model._gates(key,j,control_input)
        raw,af,sa,sb=paired_snapshots(model,q,key,slot,value,gates,old)
        ast=torch.cat((snap.inner[:,None],sa[:,:-1].to(snap.inner.dtype)),1)
        bst=torch.cat((snap.outer[:,None],sb[:,:-1].to(snap.outer.dtype)),1)
        write=model._assignment(slot)
        if write is None:
            write=torch.einsum("bnchd,bnhdm->bnchm",unit(key).to(ast.dtype).reshape(b,n,c,h,d),ast).softmax(-1).reshape(b,t,h,d)
        write=write.to(q.dtype)
        count=old.count[:,None]+write.to(old.count.dtype).cumsum(1)
        slots.append(slot);values.append(value);writes.append(write);counts.append(count)
        bases.append(model._base_logits(raw,count,None,j));afinals.append(af)
        gos.append(gates[2]);bos.append(gates[3])
        pw=write.to(ast.dtype).reshape(b,n,c,h,d)
        rk=torch.einsum("bnchm,bnhdm->bnchd",pw,ast).reshape(b,t,h,d)
        rv=torch.einsum("bnchm,bnhmv->bnchv",pw,bst).reshape(b,t,h,vd)
        key,value=model._residual(key,value,rk,rv,j)
    return read_and_output(model,q,q,torch.stack(bases,2),torch.stack(writes,2),
        pack_sequence(slots),pack_sequence(values),pack_sequence(gos),pack_sequence(bos),
        state,torch.stack(afinals,1),counts,frozen=frozen)
