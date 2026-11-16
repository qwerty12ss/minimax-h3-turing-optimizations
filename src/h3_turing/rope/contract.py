"""Operator geometry and fixture contracts; GPU qualification is local."""
S, H, D, ROT = (29226, 28, 128, 96)

CHUNKS = ((0, 8192), (8192, 8192), (16384, 8192), (24576, 4650))

PAIRS = (('float16', 'float16'), ('float16', 'float32'), ('float32', 'float16'), ('float32', 'float32'))

def require(value, message):
    if not value:
        raise ValueError(message)

def cases():
    plan = []
    for chunk, (start, count) in enumerate(CHUNKS):
        for fd, nd in PAIRS:
            for fixture in ('random_rms', 'signed_edges', 'all_half_codes'):
                plan.append({'id': f'{fixture}-{chunk}-{fd}-{nd}', 'start': start, 'count': count, 'frequency_dtype': fd, 'normal_dtype': nd, 'rot_dim': ROT, 'fixture': fixture, 'epsilon': 1e-05, **({'rms_input_dtype': nd, 'rms_weight_dtype': nd} if fixture == 'random_rms' else {})})
    for chunk in (0, 3):
        for fd, nd in PAIRS:
            for fixture in ('rf_identity_edges', 'rf_quarterturn_edges', 'rf_signedzero_edges'):
                start, count = CHUNKS[chunk]
                plan.append({'id': f'{fixture}-{chunk}-{fd}-{nd}', 'start': start, 'count': count, 'frequency_dtype': fd, 'normal_dtype': nd, 'rot_dim': ROT, 'fixture': fixture, 'epsilon': 1e-05})
        for fd in ('float16', 'float32'):
            start, count = CHUNKS[chunk]
            plan.append({'id': f'float32_payloads-{chunk}-{fd}', 'start': start, 'count': count, 'frequency_dtype': fd, 'normal_dtype': 'float32', 'rot_dim': ROT, 'fixture': 'float32_payloads', 'epsilon': 1e-06})
    return plan

def metric(value):
    require(value.get('elements') == H * S * D and value.get('mismatches') == 0 and (value.get('bitwise_equal') is True) and (value.get('signed_zero_aware') is True) and (value.get('nan_payload_aware') is True), 'Complete raw FP16 bit equality required')

def ptx_proof(value):
    require(value.get('non_ftz') is True and value.get('mul_rn_f32', 0) > 0 and (value.get('fma_rn_f32', 0) > 0) and (value.get('cvt_rn_f16_f32', 0) > 0) and (value.get('no_reduction') is True) and (value.get('target_sm75') is True) and (len(value.get('ptx_sha256', '')) == 64) and (value.get('ptx_bytes', 0) > 0), 'Actual explicit non-FTZ compiled PTX missing')

CORE_SHA = 'e7c45a7f44d4ce81c9b20786a5d5eb838ad89c284859f000a65fa4a3fbcac4a5'
