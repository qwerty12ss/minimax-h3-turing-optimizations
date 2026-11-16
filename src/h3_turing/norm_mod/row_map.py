"""Exact indexing preparation; caller resets cache at each 50-block forward."""
from .contract import require


def describe_segments(segments, count, table_rows, tensor_type):
    require(isinstance(segments, (list, tuple)) and segments and 1 <= table_rows <= 4096, 'Invalid segment/table contract')
    cursor, descriptors = 0, []
    for segment in segments:
        require(isinstance(segment, (tuple, list)) and len(segment) == 3, 'Invalid segment tuple')
        start, stop, row = segment
        require(type(start) is int and type(stop) is int and start == cursor and start < stop <= count,
                'Segments must form a disjoint exact contiguous cover')
        if type(row) is int:
            require(-table_rows <= row < table_rows, 'Integer modulation row out of bounds')
            descriptor = {'kind': 'int', 'start': start, 'stop': stop, 'row': row}
        else:
            require(isinstance(row, tensor_type) and row.ndim == 1 and tuple(row.shape) == (stop - start,),
                    'Per-token modulation row tensor must match its segment length')
            descriptor = {'kind': 'tensor', 'start': start, 'stop': stop, 'shape': list(row.shape),
                          'dtype': str(row.dtype), 'device': str(row.device), 'stride': list(row.stride())}
        descriptors.append(descriptor)
        cursor = stop
    require(cursor == count, 'Segments do not cover every residual row')
    return descriptors


class RowMapScope:
    """One source-guarded H3 forward, with an immutable mod_segments object.

    The guarded model builds segments once and passes the same list to all50
    blocks; none of those blocks mutate it. Reset before every block0 call.
    Strong references live only until the next forward, avoiding inference
    tensor version-counter assumptions and allocator/Python id reuse.
    """
    def __init__(self):
        self.segments = None
        self.maps = {}
        self.descriptors = None

    def begin(self, segments):
        self.segments = segments
        self.maps.clear()
        self.descriptors = None

    def prepare(self, segments, count, table_rows, device, torch):
        require(segments is self.segments, 'mod_segments object changed within the guarded 50-block forward')
        key = (int(count), int(table_rows), str(device))
        if key in self.maps:
            return self.maps[key]
        descriptors = describe_segments(segments, count, table_rows, torch.Tensor)
        raw = torch.empty((count,), dtype=torch.int64, device=device)
        for start, stop, row in segments:
            if type(row) is int:
                raw[start:stop].fill_(row)
            else:
                require(row.dtype is torch.int64 and row.device.type in ('cpu', 'cuda')
                        and (row.device.type == 'cpu' or row.device == device), 'Expected CPU/current-device LongTensor indices')
                raw[start:stop].copy_(row.to(device=device, dtype=torch.int64))
        # Original Torch indexing provides the same negative-index semantics and
        # out-of-range device assertion as _mod_row. No host min/max/synchronize.
        lookup = torch.arange(table_rows, dtype=torch.int64, device=device)
        mapping = lookup[raw]
        self.maps[key] = mapping
        self.descriptors = descriptors
        return mapping
