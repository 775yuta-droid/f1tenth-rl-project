"""
TFEventsファイルを依存ライブラリなしで読み取るスクリプト
"""
import struct
import sys
import zlib

def masked_crc32c(data):
    crc = zlib.crc32(data) & 0xffffffff
    return (((crc >> 15) | (crc << 17)) & 0xffffffff + 0xa282ead8) & 0xffffffff

def read_tfevents(path):
    results = {}
    with open(path, 'rb') as f:
        while True:
            # length header (8 bytes) + crc (4 bytes) = 12 bytes
            len_bytes = f.read(8)
            if len(len_bytes) < 8:
                break
            data_len = struct.unpack('<Q', len_bytes)[0]
            f.read(4)  # masked_crc32 of length
            data = f.read(data_len)
            if len(data) < data_len:
                break
            f.read(4)  # masked_crc32 of data

            # Protobuf parse (Event message):
            # Field 1 (wall_time) = double, Field 2 (step) = int64
            # Field 5 (summary) = message {Field 1 (value) = repeated message {Field 1 (tag) = string, Field 2 (simple_value) = float}}
            step = 0
            idx = 0
            while idx < len(data):
                # read tag+wire
                if idx >= len(data): break
                tw = data[idx]; idx += 1
                field = tw >> 3
                wire = tw & 0x7
                if field == 2 and wire == 0:
                    # step (varint)
                    val = 0; shift = 0
                    while True:
                        b = data[idx]; idx += 1
                        val |= (b & 0x7F) << shift
                        shift += 7
                        if not (b & 0x80): break
                    step = val
                elif field == 5 and wire == 2:
                    # summary (length-delimited)
                    slen = 0; shift = 0
                    while True:
                        b = data[idx]; idx += 1
                        slen |= (b & 0x7F) << shift
                        shift += 7
                        if not (b & 0x80): break
                    summary_data = data[idx:idx+slen]; idx += slen
                    # parse summary
                    sidx = 0
                    while sidx < len(summary_data):
                        stw = summary_data[sidx]; sidx += 1
                        sfield = stw >> 3
                        swire = stw & 0x7
                        if sfield == 1 and swire == 2:
                            # value message
                            vlen = 0; shift = 0
                            while True:
                                b = summary_data[sidx]; sidx += 1
                                vlen |= (b & 0x7F) << shift
                                shift += 7
                                if not (b & 0x80): break
                            value_data = summary_data[sidx:sidx+vlen]; sidx += vlen
                            # parse value: field 1=tag(string), field 2=simple_value(float)
                            tag = None; simple_value = None
                            vidx = 0
                            while vidx < len(value_data):
                                vtw = value_data[vidx]; vidx += 1
                                vfield = vtw >> 3
                                vwire = vtw & 0x7
                                if vfield == 1 and vwire == 2:
                                    tlen = 0; shift = 0
                                    while True:
                                        b = value_data[vidx]; vidx += 1
                                        tlen |= (b & 0x7F) << shift
                                        shift += 7
                                        if not (b & 0x80): break
                                    tag = value_data[vidx:vidx+tlen].decode('utf-8'); vidx += tlen
                                elif vfield == 2 and vwire == 5:
                                    simple_value = struct.unpack('<f', value_data[vidx:vidx+4])[0]; vidx += 4
                                else:
                                    break
                            if tag and simple_value is not None:
                                if tag not in results:
                                    results[tag] = []
                                results[tag].append((step, simple_value))
                        else:
                            # skip unknown
                            if swire == 0:
                                while True:
                                    b = summary_data[sidx]; sidx += 1
                                    if not (b & 0x80): break
                            elif swire == 2:
                                slen2 = 0; shift = 0
                                while True:
                                    b = summary_data[sidx]; sidx += 1
                                    slen2 |= (b & 0x7F) << shift
                                    shift += 7
                                    if not (b & 0x80): break
                                sidx += slen2
                            else:
                                break
                else:
                    # skip
                    if wire == 0:
                        while idx < len(data):
                            b = data[idx]; idx += 1
                            if not (b & 0x80): break
                    elif wire == 1:
                        idx += 8
                    elif wire == 2:
                        slen = 0; shift = 0
                        while True:
                            b = data[idx]; idx += 1
                            slen |= (b & 0x7F) << shift
                            shift += 7
                            if not (b & 0x80): break
                        idx += slen
                    elif wire == 5:
                        idx += 4
                    else:
                        break
    return results

if __name__ == '__main__':
    path = '/home/yuta775/projects/f1tenth-rl-project/logs/PPO_17/events.out.tfevents.1771927753.MSI.1223.0'
    results = read_tfevents(path)
    print(f"取得できたメトリクス数: {len(results)}")
    for tag in sorted(results.keys()):
        vals = results[tag]
        vals_v = [v for _, v in vals]
        print(f"\n【{tag}】")
        print(f"  記録ステップ数: {len(vals)}")
        print(f"  初期値: {vals[0][1]:.4f} (step={vals[0][0]})")
        print(f"  最終値: {vals[-1][1]:.4f} (step={vals[-1][0]})")
        print(f"  最大値: {max(vals_v):.4f}")
        print(f"  最小値: {min(vals_v):.4f}")
