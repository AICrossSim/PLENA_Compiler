#!/usr/bin/env python3
"""Supply-v3 front end: physical MX payloads and compact, byte-exact tile layouts.

The legacy compiler is untouched. Timing is produced by the Rust event simulator,
not by this front end. --reference-dir permits standalone checkout use.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_REFERENCE=HERE.parents[2]/'moe-supply-first-v3-simulator/research/moe_dispatch/v3_reference'


def _reference(name):
    candidates=[Path(os.environ['PLENA_V3_REFERENCE_DIR'])] if os.environ.get('PLENA_V3_REFERENCE_DIR') else []
    candidates += [HERE/'v3_reference',DEFAULT_REFERENCE]
    for p in candidates:
        src=p/(name+'.py')
        if src.exists():
            spec=importlib.util.spec_from_file_location('plena_v3_'+name,src); m=importlib.util.module_from_spec(spec)
            sys.modules[spec.name]=m; spec.loader.exec_module(m); return m
    raise FileNotFoundError('v3 reference not found; set PLENA_V3_REFERENCE_DIR')

if '--reference-dir' in sys.argv:
    os.environ['PLENA_V3_REFERENCE_DIR']=sys.argv[sys.argv.index('--reference-dir')+1]

bv=_reference('budget_v3'); rn=_reference('ref_numerics')


def _aligned(raw,granule=32):
    return raw+b'\x00'*((-len(raw))%granule)


def pack_codes(codes,bits):
    """Little-endian two's-complement bit stream (supports 3-bit codes)."""
    q=np.asarray(codes,np.int16).reshape(-1)
    if not 2<=bits<=8 or np.any(q < -(1<<(bits-1))) or np.any(q>(1<<(bits-1))-1):
        raise ValueError('code outside signed bitwidth')
    u=(q & ((1<<bits)-1)).astype(np.uint16)
    bit=((u[:,None] >> np.arange(bits))&1).astype(np.uint8).reshape(-1)
    return np.packbits(bit,bitorder='little').tobytes()


def unpack_codes(raw,bits,shape):
    n=math.prod(shape); bit=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')[:n*bits].reshape(n,bits)
    u=(bit.astype(np.int16)*(1<<np.arange(bits))).sum(1)
    q=np.where(u >= (1<<(bits-1)),u-(1<<bits),u).astype(np.int8)
    return q.reshape(shape)


def bf16_bytes(x):
    return (rn.bf16_round(x).view(np.uint32)>>16).astype('<u2').tobytes()


def pack_mx_tile(w,bits=4,split8=False):
    w=np.asarray(w,np.float32)
    if w.ndim!=2: raise ValueError('MX tile is output rows x reduction')
    if split8:
        hi,lo,e,deq=rn.mxint8_split(w,red_axis=1)
        q=(hi.astype(np.int16)*16+lo.astype(np.int16)).astype(np.int8)
    else: q,e,deq=rn.mx_quantize(w,bits)
    code=pack_codes(q,bits); ex=(e+127).astype(np.uint8).tobytes()
    raw=_aligned(code+ex)
    return raw,{'shape_nk':list(w.shape),'bits':bits,'code_bytes':len(code),'scale_bytes':len(ex),
                'bytes':len(raw),'scale_offset':len(code),'dequantized':deq}


def unpack_mx_tile(raw,meta):
    shape=meta['shape_nk']; bits=meta['bits']
    q=unpack_codes(raw[:meta['code_bytes']],bits,shape).astype(np.float32)
    ex=np.frombuffer(raw[meta['scale_offset']:meta['scale_offset']+meta['scale_bytes']],np.uint8).astype(np.int16)-127
    ex=ex.reshape(shape[0],-(-shape[1]//32))
    return q*np.repeat(np.exp2(ex.astype(np.float32)),32,axis=1)[:,:shape[1]]


def pack_factor(f,fmt):
    """Pack reduction x four output columns; MX B blocks never cross rank sets."""
    f=np.asarray(f,np.float32)
    if fmt=='bf16': return bf16_bytes(f),rn.bf16_round(f)
    if fmt not in ('mxint4','mxint8','mxint8s'): raise ValueError(fmt)
    bits=4 if fmt=='mxint4' else 8
    if fmt=='mxint8s':
        hi,lo,e,deq=rn.mxint8_split(f,red_axis=0)
        q=(16*hi.astype(np.int16)+lo.astype(np.int16)).astype(np.int8)
    else: q,e,deq=rn.mx_quantize(f,bits,axis=0)
    # Store each output column's reduction stream; exponent table follows codes.
    raw=pack_codes(q.T,bits)+(e+127).astype(np.uint8).tobytes()
    return raw,deq


def rank_layout(r,K,L,placement='interleave',streamed=False,slack_pack=False):
    S=bv.nseg(K); capacities=[L]*S
    if slack_pack: capacities[-1]+=S*512-K
    cap=(capacities[-1] if streamed else sum(capacities)); fused=min(r,cap)
    seg=[[] for _ in range(S)]
    if streamed: seg[-1]=list(range(fused))
    elif fused: seg=rn.rank_segments(fused,K,L,placement,slack_pack)
    tails=[list(range(i,min(r,i+max(1,L)))) for i in range(fused,r,max(1,L))]
    assert all(len(R)<=c for R,c in zip(seg,capacities))
    return seg,tails


def projection_layout(name,K,N,r,fmt,base=0,streamed=False):
    """One descriptor per K segment; N columns expand deterministically at runtime."""
    p=bv.plan_projection(name,K,N,r,fmt)
    segs,tails=rank_layout(p.r,K,fmt.L,streamed=streamed,slack_pack=fmt.slack_pack) if p.r and fmt.comp=='lanes' else ([[] for _ in range(bv.nseg(K))],[])
    u_order=[j for group in segs+tails for j in group] if fmt.comp=='lanes' else list(range(p.r))
    assert sorted(u_order)==list(range(p.r))
    u_inverse=[0]*p.r
    for physical,logical in enumerate(u_order):u_inverse[logical]=physical
    u_offsets=[];cursor=0
    for group in segs:u_offsets.append({'rank_start':cursor,'rank_count':len(group)});cursor+=len(group)
    at=base; blocks=[]
    if fmt.comp=='kext':
        sizes=p.seg_tile_bytes
    else:
        sizes=[bv.align(bv.main_tile_bytes(bv.seg_len(K,s),fmt.main_bits)+(bv.factor_block_bytes(len(R),4,fmt.factor) if R else 0)) for s,R in enumerate(segs)]
    for s,size in enumerate(sizes):
        blocks.append({'kind':'main','k_segment':s,'k_start':s*512,'k_len':max(0,min(512,K-s*512)),
                       'n_tiles':bv.cdiv(N,4),'n_tile':4,'tile_bytes':size,'hbm_base':at,'rank_indices':segs[s] if s<len(segs) else [],
                       'requests_32b':bv.cdiv(size,32),'pool_reads':1})
        at+=bv.cdiv(N,4)*size
    a=[]
    if p.r:
        for s in range(bv.nseg(K)):
            size=bv.align(bv.factor_block_bytes(bv.seg_len(K,s),4,bv.a_fmt(fmt)))
            a.append({'kind':'a_prepass','k_segment':s,'k_len':bv.seg_len(K,s),'n_tiles':bv.cdiv(p.r,4),
                      'tile_bytes':size,'hbm_base':at,'int4_passes':bv.a_passes(fmt)})
            at+=bv.cdiv(p.r,4)*size
    tail=[]
    if p.r_tail or (streamed and tails):
        # B_tail stored once per N tile. Several lane-only issues read the same block.
        nr=sum(map(len,tails)) if streamed else p.r_tail
        size=bv.align(bv.factor_block_bytes(nr,4,fmt.factor))
        tail=[{'kind':'b_tail','n_tiles':bv.cdiv(N,4),'tile_bytes':size,'hbm_base':at,'rank_indices':list(range(p.r-nr,p.r)),
               'lane_only_issues_per_n':bv.cdiv(nr,max(1,fmt.L)) if fmt.comp=='lanes' else bv.cdiv(nr,512)}]
        at+=bv.cdiv(N,4)*size
    return {'phase':name,'K':K,'N':N,'rank':p.r,'main_segments':blocks,'a_segments':a,'tail_segments':tail,
            'a_prepass_loop_order':{'WS':['rank_group','K_segment','rank_tile'],'IS':['K_segment','rank_tile']},
            'a_rank_group_columns':32,'a_physical_layout':'K_segment_major; AGU computes rank-group offsets without re-fetch',
            'u_bf16_layout':{'physical_to_logical_rank':u_order,'logical_to_physical_rank':u_inverse,
                             'segment_slices':u_offsets,'tail_rank_start':cursor,
                             'partial_fp32_order':'logical rank; UStore permutes into BF16 layout'},
            'down_combine_schedule':('K_segment then N_group: atomic partial drain after each main segment; lane-only rank tails likewise; no private Me-by-H Y' if name=='down' and streamed else 'final output N_group drain after ascending-K accumulation'),
            'main_tiles':sum(v['n_tiles'] for v in blocks),'main_bytes':sum(v['n_tiles']*v['tile_bytes'] for v in blocks),
            'a_bytes':sum(v['n_tiles']*v['tile_bytes'] for v in a),'b_tail_bytes':sum(v['n_tiles']*v['tile_bytes'] for v in tail),
            'hbm_base':base,'end_address':at,'bytes':at-base,'expected_budget_bytes':p.main_bytes+p.a_bytes+p.b_sep_bytes}


def pack_projection(W,A,B,fmt,name='projection',streamed=False):
    """Materialize an immutable main/A/B payload; labels contain actual physical offsets."""
    W=np.asarray(W,np.float32); N,K=W.shape; A=np.asarray(A,np.float32); B=np.asarray(B,np.float32); r=A.shape[1]
    if A.shape!=(K,r) or B.shape!=(r,N): raise ValueError('factor shapes disagree')
    if fmt.comp=='kext': raise ValueError('kext packed payload is generated by extending decoded P1 operands; use projection_layout for its transfer descriptors')
    p=bv.plan_projection(name,K,N,r,fmt); r=p.r
    segs,tails=rank_layout(r,K,fmt.L,streamed=streamed,slack_pack=fmt.slack_pack) if r and fmt.comp=='lanes' else ([[] for _ in range(bv.nseg(K))],[])
    if r and fmt.comp!='lanes': tails=[list(range(r))]
    raw=bytearray(); records=[]
    def emit(kind,s,n,part,size,extra):
        if len(part)>size: raise AssertionError((kind,len(part),size))
        records.append({'kind':kind,'k_segment':s,'n_start':n,'offset':len(raw),'bytes':size,**extra})
        raw.extend(part); raw.extend(b'\x00'*(size-len(part)))
    for s,R in enumerate(segs):
        kv=bv.seg_len(K,s)
        for n in range(0,N,4):
            tile=np.zeros((4,kv),np.float32);tile[:min(4,N-n)]=W[n:n+4,s*512:s*512+kv]
            main=bf16_bytes(tile) if fmt.main_bits==16 else pack_mx_tile(tile,fmt.main_bits)[0]
            bf=np.zeros((len(R),4),np.float32)
            if R: bf[:,:min(4,N-n)]=B[R,n:n+4]
            b=pack_factor(bf,fmt.factor)[0] if R else b''
            size=bv.align(len(main)+len(b)); emit('main',s,n,main+b,size,{'rank_indices':R,'main_bytes':len(main),'b_bytes':len(b)})
    if r:
        for s in range(bv.nseg(K)):
            kv=bv.seg_len(K,s)
            for n in range(0,r,4):
                af=np.zeros((kv,4),np.float32);af[:,:min(4,r-n)]=A[s*512:s*512+kv,n:n+4]
                payload=bf16_bytes(af.T) if bv.a_fmt(fmt)=='bf16' else pack_factor(af,bv.a_fmt(fmt))[0];size=bv.align(bv.factor_block_bytes(kv,4,bv.a_fmt(fmt)))
                emit('a_prepass',s,n,payload,size,{'int4_passes':bv.a_passes(fmt),'factor_layout':'output_major','factor_shape':[kv,4]})
        remaining=[j for R in tails for j in R]
        if remaining:
            for n in range(0,N,4):
                bf=np.zeros((len(remaining),4),np.float32);bf[:,:min(4,N-n)]=B[remaining,n:n+4]
                payload,_=pack_factor(bf,fmt.factor);size=bv.align(bv.factor_block_bytes(len(remaining),4,fmt.factor))
                emit('b_tail',-1,n,payload,size,{'rank_indices':remaining})
    expected=projection_layout(name,K,N,r,fmt,streamed=streamed)['bytes']
    assert len(raw)==expected,(len(raw),expected)
    return bytes(raw),{'schema':'plena_moe_v3_payload_v1','phase':name,'shape_nk':[N,K],'rank':r,'bytes':len(raw),
                       'sha256':hashlib.sha256(raw).hexdigest(),'tiles':records,'hbm_endianness':'little','code_layout':'output_major','b_bf16_layout':'rank_major'}


def unpack_projection(raw,metadata,fmt):
    """Decode the physical payload for numeric replay; no original floats needed."""
    N,K=metadata['shape_nk'];r=metadata['rank']
    W=np.zeros((N,K),np.float32);A=np.zeros((K,r),np.float32);B=np.zeros((r,N),np.float32)
    def unbf(buf,shape):
        n=math.prod(shape);u=np.frombuffer(buf[:2*n],dtype='<u2').astype(np.uint32)<<16
        return u.view(np.float32).reshape(shape)
    def unmx(buf,nrows,red,bits):
        code=(nrows*red*bits+7)//8;scales=nrows*bv.cdiv(red,32)
        return unpack_mx_tile(buf,{'shape_nk':[nrows,red],'bits':bits,'code_bytes':code,'scale_bytes':scales,'scale_offset':code})
    def unb(buf,nred,ncol,factor):
        if factor=='bf16':return unbf(buf,(nred,ncol))
        return unmx(buf,ncol,nred,4 if factor=='mxint4' else 8).T
    for tile in metadata['tiles']:
        n=tile['n_start'];cols=min(4,(r if tile['kind']=='a_prepass' else N)-n);s=tile['k_segment'];buf=raw[tile['offset']:tile['offset']+tile['bytes']]
        if tile['kind']=='main':
            kv=bv.seg_len(K,s);main=unbf(buf,(4,kv)) if fmt.main_bits==16 else unmx(buf,4,kv,fmt.main_bits)
            W[n:n+cols,s*512:s*512+kv]=main[:cols]
            ranks=tile['rank_indices']
            if ranks:B[np.ix_(ranks,range(n,n+cols))]=unb(buf[tile['main_bytes']:],len(ranks),4,fmt.factor)[:,:cols]
        elif tile['kind']=='a_prepass':
            kv=bv.seg_len(K,s);f=unbf(buf,(4,kv)).T if bv.a_fmt(fmt)=='bf16' else unmx(buf,4,kv,4 if bv.a_fmt(fmt)=='mxint4' else 8).T
            A[s*512:s*512+kv,n:n+cols]=f[:,:cols]
        elif tile['kind']=='b_tail':
            ranks=tile['rank_indices'];B[np.ix_(ranks,range(n,n+cols))]=unb(buf,len(ranks),4,fmt.factor)[:,:cols]
    return W,A,B


def unpack_a8_passes(raw,metadata):
    """Derive scaled high/low INT4 pass operands from the actual packed MXINT8 A.

    This does not quantize source floats again. Both returned matrices are
    K×rank; high uses exponent e+4 and low uses e, including low code -8.
    """
    N,K=metadata['shape_nk'];r=metadata['rank'];high=np.zeros((K,r),np.float32);low=np.zeros_like(high)
    for tile in metadata['tiles']:
        if tile['kind']!='a_prepass':continue
        s,n=tile['k_segment'],tile['n_start'];kv=bv.seg_len(K,s);cols=min(4,r-n)
        buf=raw[tile['offset']:tile['offset']+tile['bytes']];codes=unpack_codes(buf[:4*kv],8,(4,kv)).astype(np.int16)
        count=4*bv.cdiv(kv,32);e=np.frombuffer(buf[4*kv:4*kv+count],np.uint8).astype(np.int16).reshape(4,-1)-127
        scale=np.repeat(np.exp2(e.astype(np.float32)),32,axis=1)[:,:kv]
        hi=(codes+8)//16;lo=codes-16*hi
        if np.any(hi<-7) or np.any(hi>7) or np.any(lo<-8) or np.any(lo>7):raise ValueError('illegal MXINT8-split code')
        high[s*512:s*512+kv,n:n+cols]=(hi.astype(np.float32)*16*scale)[:cols].T
        low[s*512:s*512+kv,n:n+cols]=(lo.astype(np.float32)*scale)[:cols].T
    return high,low


def build_v3_expert(H,F,Me,shared=False,fmt=None,z_mode='full'):
    fmt=fmt or bv.FMT_V3
    ranks=bv.DEFAULT_RANKS[fmt.L if fmt.L in (8,16) else 8]['shared' if shared else 'routed']
    at=0; phases=[]
    for (name,K,N),r in zip([('gate',H,F),('up',H,F),('down',F,H)],ranks):
        p=projection_layout(name,K,N,r,fmt,at,streamed=z_mode=='streamed' and name=='down');phases.append(p);at=p['end_address']
    return {'Me':Me,'H':H,'F':F,'is_shared':shared,'z_mode':z_mode,'phases':phases,'bytes':at,
            'bf16_bytes':6*H*F,'main_tiles':sum(p['main_tiles'] for p in phases),
            'rank_lanes':fmt.L,'factor_a':bv.a_fmt(fmt),'factor_b':fmt.factor,'main_bits':fmt.main_bits}


def organization_storage(dims,T,lanes=(4,2),precision="P2",L=8,specialized=True):
    """Actual capacity by organization; global Z/U pools use admission credits.

    Two contexts share the array, WOR, XOR and accumulator. They may interleave
    compute only when both live accumulator footprints fit the same arena and
    their Z/U and WOR reservations are simultaneously legal. Each context owns
    disjoint accumulator addresses. Otherwise Next remains prefetch-only.
    Generic cores switch modes in backing storage reserved as max(WS,IS).
    Z/U globals are quota shared: two tasks cannot both exceed their available
    combined allocation, regardless of how many logical contexts are queued.
    """
    ref=bv.storage_plan(dims,T,precision,L)
    if len(lanes)==2 and lanes[0]!=lanes[1] and specialized:
        ref['structures']['dense_X_register']=2*lanes[0]*1024
        ref['total']=sum(ref['structures'].values());ref['slack']=bv.CAPACITY_BYTES-ref['total'];ref['fits']=ref['slack']>=0
        ref['organization']=list(lanes);ref['per_core']=[{'M':lanes[0],'contexts':1,'is_max_me':0,'X_bytes':ref['structures']['dense_X_register'],'acc_bytes':ref['structures']['dense_acc_buffer']},
            {'M':lanes[1],'contexts':1,'is_max_me':4,'X_bytes':ref['structures']['stream_X_register'],'acc_bytes':ref['structures']['stream_acc_sram'],
             'X_contract':'two whole-K-slice buffers, each up to4 valid rows, independent of physical M; multiple row waves reuse the same resident slice'}]
        ref['budget_contract']='frozen specialized 4+2 reference';return ref
    structs=dict(ref['structures']);tile=bv.align(bv.main_tile_bytes(512,4)+bv.factor_block_bytes(L,4,"bf16"))
    wslot=tile if precision=="P2" else 4096
    # All organizations get exactly the specialized design's eighteen WOR slots.
    structs.pop('dense_W_register');structs.pop('stream_W_register');structs['WOR_total']=18*wslot
    structs.pop('dense_X_register');structs.pop('stream_X_register')
    structs.pop('dense_acc_buffer');structs.pop('stream_acc_sram')
    per=[]
    for i,M in enumerate(lanes):
        is_cap=M
        xb=2*M*1024
        ws=2*T*32*4
        isb=bv.align(is_cap*max(2*dims.I_r,dims.d)*4,4096)
        ab=max(ws,isb)
        structs[f'core{i}_X_register']=xb;structs[f'core{i}_acc_mode_shared']=ab
        per.append({'M':M,'contexts':2 if len(lanes)==1 else 1,'is_max_me_routed':is_cap,'X_bytes':xb,'acc_bytes':ab,
                    'is_legality':'Me <= M and Me * max(2F,H) * 4 <= acc_bytes; otherwise WS',
                    'context_acc_admission':'sum(live WS Me*32*4, live IS Me*max(2F,H)*4) <= acc_bytes; disjoint addresses',
                    'acc_backing':'max(ws RF capacity, is SRAM capacity), switch charged by simulator'})
    total=sum(structs.values())
    return {'structures':structs,'total':total,'capacity':bv.CAPACITY_BYTES,'fits':total<=bv.CAPACITY_BYTES,
            'slack':bv.CAPACITY_BYTES-total,'z_mode':ref['z_mode'],'organization':list(lanes),'per_core':per,
            'budget_contract':'shared bounded Z/U quotas; Current/Next share operand/accumulator storage; eighteen WOR slots aggregate'}


def compile_v3(workload,lanes=(4,2),precision='P2',fmt=None,iso_port=True,t_chunk=None,bandwidth=256,dataflow='specialized'):
    fmt=fmt or (bv.BF16 if precision=='P0' else bv.FMT_V3)
    if precision=='P2' and (bv.a_fmt(fmt)=='bf16' or fmt.comp=='kext' or fmt.slack_pack or (fmt.comp in ('separate','offload') and fmt.factor!='mxint4')): raise ValueError('format requires BF16 P1')
    if tuple(lanes) not in ((6,),(3,3),(4,2),(8,),(4,4),(6,2),(5,3)): raise ValueError('unsupported organization')
    cases=json.loads(json.dumps(workload)); T=cases.get('batch',cases.get('t_chunk',0)); experts=cases['experts']
    shared=next(e for e in experts if e.get('is_shared',False)); d=shared['H']; Ir=next((e['F'] for e in experts if not e.get('is_shared')),shared['F']//2)
    dims=bv.Dims('captured',d,Ir,shared['F'],64,cases.get('top_k',6))
    # Physical storage frozen at worst-case whole chunk, never sized to winning dispatch.
    t_capacity=t_chunk or (128 if precision=='P2' and fmt.L==8 and bandwidth!=512 else 96)
    if T>t_capacity: raise ValueError('task exceeds frozen hardware t_chunk')
    storage=organization_storage(dims,t_capacity,lanes,precision,fmt.L or 8,dataflow=='specialized' and tuple(lanes) in ((4,2),(6,2),(5,3)))
    live_storage=organization_storage(dims,T,lanes,precision,fmt.L or 8,dataflow=='specialized' and tuple(lanes) in ((4,2),(6,2),(5,3)))
    if bandwidth==512:
        for state in (storage,live_storage):
            state['structures']['landing_pool']=128*1024;state['total']=sum(state['structures'].values());state['slack']=bv.CAPACITY_BYTES-state['total'];state['fits']=state['slack']>=0
    if not storage['fits']: raise ValueError(f"SRAM budget exceeded by {-storage['slack']} bytes")
    records=[]; at=0
    for e in sorted(experts,key=lambda x:(not x.get('is_shared',False),x['id'])):
        z_budget=storage['structures']['Z_dense'] if e.get('is_shared') else storage['structures']['Z_dense']+storage['structures']['Z_stream']
        z='full' if e['Me']*e['F']*2 <= z_budget else 'streamed'
        engine=build_v3_expert(e['H'],e['F'],e['Me'],e.get('is_shared',False),fmt,z)
        engine['hbm_base']=at;engine['expert_id']=e['id'];at+=engine['bytes'];records.append(engine)
    cases.update({'arch':'supply_v3','v3_engine_layout':records,'storage':storage,'task_live_storage':live_storage,
                  'byte_stats':{'unique_weight_bytes':at,'bf16_weight_bytes':sum(v['bf16_bytes'] for v in records),
                                'main_tiles':sum(v['main_tiles'] for v in records)},
                  'v3_config':{'precision':precision,'main_bits':fmt.main_bits,'factor_a':bv.a_fmt(fmt),'factor_b':fmt.factor,
                               'rank_lanes':fmt.L,'comp_mode':fmt.comp,'lanes':list(lanes),'iso_port':iso_port,
                               'z_mode':'auto','physical_z_mode':storage['z_mode'],'t_chunk':t_capacity,'bandwidth':bandwidth,'dataflow_mode':dataflow,'pool_bytes':storage['structures']['landing_pool'],'pool_read_total':1024,'x_port_total':640}})
    return cases


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reference-dir',type=Path);ap.add_argument('--workload-json',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--arch',choices=['supply_v3'],default='supply_v3');ap.add_argument('--lanes',default='4,2');ap.add_argument('--precision',choices=['P0','P1','P2'],default='P2')
    ap.add_argument('--t-chunk',type=int);ap.add_argument('--bandwidth',type=int,choices=[128,256,512],default=256);ap.add_argument('--dataflow',choices=['specialized','switchable'],default='specialized');ap.add_argument('--comp-mode',choices=['none','lanes','separate','kext','offload'],default='lanes');ap.add_argument('--slack-pack',action='store_true');ap.add_argument('--main-bits',type=int,choices=[3,4],default=4);ap.add_argument('--factor-a',choices=['mxint4','mxint8s','bf16'],default='mxint4')
    ap.add_argument('--factor-b',choices=['bf16','mxint8','mxint4'],default='bf16');ap.add_argument('--rank-lanes',type=int,choices=[8,16],default=8)
    args=ap.parse_args(); data=json.loads(args.workload_json.read_text()); inputs=data.get('workloads',[data]);fmt=bv.BF16 if args.precision=='P0' else bv.Fmt('v3',args.main_bits,args.factor_b,args.rank_lanes,args.factor_a,args.comp_mode,args.slack_pack)
    out={'schema':'plena_moe_supply_v3_plan_v1','arch':'supply_v3','provenance':data.get('provenance',{}),
         'workloads':[compile_v3(w,tuple(map(int,args.lanes.split(','))),args.precision,fmt,t_chunk=args.t_chunk,bandwidth=args.bandwidth,dataflow=args.dataflow) for w in inputs]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
