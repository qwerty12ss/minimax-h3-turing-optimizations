"""Operator shape and launch contracts; publication has separate GPU tests."""
SHAPE = (29226, 5376)

CONFIGURATIONS = ((256, 4), (512, 4), (1024, 4), (2048, 4), (1024, 8), (2048, 8))

def require(value, message):
    if not value:
        raise RuntimeError(message)

def validate_configuration(block, warps):
    require((int(block), int(warps)) in CONFIGURATIONS, 'Unreviewed pointwise launch configuration')
