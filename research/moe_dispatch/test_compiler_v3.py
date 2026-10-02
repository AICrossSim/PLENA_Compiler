import numpy as np
import pytest
import compiler_v3 as c


def test_exact_payload_budget():
    r=c.build_v3_expert(2048,1408,2,False);s=c.build_v3_expert(2048,2816,16,True)
    assert r['bytes']==4970112 and s['bytes']==9889920
    assert r['main_tiles']==4352 and s['main_tiles']==8704
    assert r['phases'][2]['main_segments'][-1]['tile_bytes']==896
    assert sum(p['a_segments'][s]['n_tiles'] for p in r['phases'] for s in range(len(p['a_segments'])))==82


def test_mx_roundtrip_all_bits_and_negative8():
    rng=np.random.default_rng(11);w=rng.normal(size=(4,384)).astype(np.float32)
    for bits in (2,3,4,8):
        raw,m=c.pack_mx_tile(w,bits)
        assert np.array_equal(c.unpack_mx_tile(raw,m),m['dequantized'])
    q=np.array([-8,-7,-1,0,1,7],np.int8)
    assert np.array_equal(c.unpack_codes(c.pack_codes(q,4),4,q.shape),q)


def test_actual_deepseek_payload_has_exact_bytes():
    # Real dimensions, deterministic synthetic values; this verifies physical packing, not accuracy.
    rng=np.random.default_rng(12);payload_bytes=0
    for name,K,N,r in [('gate',2048,1408,32),('up',2048,1408,32),('down',1408,2048,24)]:
        w=rng.normal(size=(N,K)).astype(np.float32)*.01;a=rng.normal(size=(K,r)).astype(np.float32)*.01;b=rng.normal(size=(r,N)).astype(np.float32)*.01
        raw,meta=c.pack_projection(w,a,b,c.bv.FMT_V3,name)
        assert len(raw)==meta['bytes']; assert all(t['offset']%32==0 and t['bytes']%32==0 for t in meta['tiles'])
        assert all(len(t.get('rank_indices',[]))<=8 for t in meta['tiles'] if t['kind']=='main')
        payload_bytes+=len(raw)
    assert payload_bytes==4970112


def test_streamed_tail_is_counted_and_factor_formats():
    rng=np.random.default_rng(13);K,N,r=1408,12,48;w=rng.normal(size=(N,K)).astype(np.float32);a=rng.normal(size=(K,r)).astype(np.float32);b=rng.normal(size=(r,N)).astype(np.float32)
    for factor_a in ('mxint4','mxint8s','bf16'):
        fmt=c.bv.Fmt('test',4,'bf16',8,factor_a=factor_a)
        raw,m=c.pack_projection(w,a,b,fmt,streamed=True)
        mains=[v for v in m['tiles'] if v['kind']=='main']
        assert all(not v['rank_indices'] for v in mains if v['k_segment']<2)
        assert all(len(v['rank_indices'])==8 for v in mains if v['k_segment']==2)
        assert any(v['kind']=='b_tail' for v in m['tiles'])
        assert len(raw)==c.projection_layout('test',K,N,r,fmt,streamed=True)['bytes']


def test_storage_by_org_and_capacity_limit():
    dims=c.bv.MODELS['dsv2_lite']
    for org in ((6,),(3,3),(4,2),(8,),(4,4),(6,2),(5,3)):
        p=c.organization_storage(dims,96,org)
        assert p['total']==sum(p['structures'].values())
        assert p['fits']
    assert not c.organization_storage(dims,128,(4,2),'P1')['fits']
    assert not c.organization_storage(dims,128,(4,2),'P2',16)['fits']


def test_projection_decodes_actual_payload_not_source_floats():
    rng=np.random.default_rng(17);N,K,r=9,544,8
    W=rng.normal(size=(N,K)).astype(np.float32)*.04;A=rng.normal(size=(K,r)).astype(np.float32)*.02;B=rng.normal(size=(r,N)).astype(np.float32)*.02
    for af in ('mxint4','mxint8s','bf16'):
        for bf in ('bf16','mxint8','mxint4'):
            fmt=c.bv.Fmt('packed',4,bf,4,af)
            raw,meta=c.pack_projection(W,A,B,fmt)
            Wq,Aq,Bq=c.unpack_projection(raw,meta,fmt)
            assert np.array_equal(Wq,c.rn.mx_quantize(W,4)[2])
            agold=c.rn.mxint8_split(A,0)[3] if af=='mxint8s' else c.rn.quantize_factor(A,af,0)
            assert np.array_equal(Aq,agold)
            bgold=c.rn.quantize_b_per_segment(B,c.rn.rank_segments(r,K,4),bf)
            assert np.array_equal(Bq,bgold)


def test_a8_two_pass_operands_come_from_packed_codes():
    rng=np.random.default_rng(31);W=rng.normal(size=(12,544)).astype(np.float32);A=rng.normal(size=(544,8)).astype(np.float32);B=rng.normal(size=(8,12)).astype(np.float32)
    fmt=c.bv.Fmt('a8_actual',4,'bf16',4,'mxint8s');raw,metadata=c.pack_projection(W,A,B,fmt)
    _,decoded,_=c.unpack_projection(raw,metadata,fmt);hi,lo=c.unpack_a8_passes(raw,metadata)
    assert np.array_equal(hi+lo,decoded)
    assert np.any(lo<0) and np.any(hi<0)


def test_u_physical_slices_preserve_logical_rank_values():
    full=c.projection_layout('down',1408,2048,24,c.bv.FMT_V3)
    u=full['u_bf16_layout'];order=u['physical_to_logical_rank'];inverse=u['logical_to_physical_rank']
    logical=np.arange(24);physical=logical[order]
    assert np.array_equal(physical[inverse],logical)
    for seg,desc in zip(full['main_segments'],u['segment_slices']):
        start,count=desc['rank_start'],desc['rank_count']
        assert order[start:start+count]==seg['rank_indices']
    streamed=c.projection_layout('down',1408,2048,24,c.bv.FMT_V3,streamed=True)
    assert streamed['u_bf16_layout']['physical_to_logical_rank']==list(range(24))
    assert streamed['u_bf16_layout']['tail_rank_start']==8
    assert 'no private' in streamed['down_combine_schedule']
