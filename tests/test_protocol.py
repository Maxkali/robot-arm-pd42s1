# -*- coding: utf-8 -*-
"""
PD42S1 协议单元测试
- 帧构建/解析往返
- 校验和正确性
- 异常帧处理
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pd42s1_driver import (
    build_frame, parse_frame, checksum,
    FRAME_HEAD, FRAME_TAIL, SerialFrame
)


def test_build_frame_structure():
    """构建帧结构: 0xC5 + addr + func + payload + cs + 0x5C"""
    frame = build_frame(addr=1, func=0xF2, payload=b'\x00\x01')
    assert frame[0] == FRAME_HEAD
    assert frame[1] == 1           # addr
    assert frame[2] == 0xF2        # func
    assert frame[3:5] == b'\x00\x01'  # payload
    assert frame[-1] == FRAME_TAIL

    # 校验和 = sum(head+addr+func+payload) & 0xFF
    expected_cs = sum(frame[:-2]) & 0xFF
    assert frame[-2] == expected_cs
    print('[OK] 帧结构构建测试通过')


def test_checksum():
    """校验和计算"""
    assert checksum(b'\xC5\x01\xF2') == (0xC5 + 0x01 + 0xF2) & 0xFF
    assert checksum(b'') == 0
    print('[OK] 校验和测试通过')


def test_parse_roundtrip():
    """构建命令帧 -> 手动加错误码转应答帧 -> 解析"""
    # build_frame 生成命令帧: 0xC5 + addr + func + payload + cs + 0x5C
    cmd = build_frame(addr=3, func=0x2A, payload=b'\x00\x01\x02\x03')

    # 应答帧比命令帧多一个错误码字节: 在 func 后插入 0x01(成功)
    resp_body = cmd[:3] + b'\x01' + cmd[3:-2]  # head+addr+func + err + payload
    resp = resp_body + bytes([checksum(resp_body), FRAME_TAIL])

    parsed = parse_frame(resp)
    assert parsed is not None
    assert parsed.slave_addr == 3
    assert parsed.function_code == 0x2A
    assert parsed.data == b'\x00\x01\x02\x03'
    assert parsed.valid is True
    assert parsed.ok() is True
    print('[OK] 帧构建->解析往返测试通过')


def test_parse_response_with_error_code():
    """应答帧含错误码: 0xC5 + addr + func + err + data + cs + 0x5C"""
    # 构造一个应答帧 (带错误码 0x01=成功)
    body = bytes([FRAME_HEAD, 1, 0x2A, 0x01]) + b'\x00\x00\x00\x00'
    resp = body + bytes([checksum(body), FRAME_TAIL])
    parsed = parse_frame(resp)
    assert parsed is not None
    assert parsed.error_code == 0x01
    assert parsed.ok() is True
    assert parsed.data == b'\x00\x00\x00\x00'
    print('[OK] 应答帧解析(含错误码)测试通过')


def test_parse_invalid_frame():
    """无效帧返回 None 或 valid=False"""
    # 太短
    assert parse_frame(b'\xC5\x01') is None
    # 帧头错误
    assert parse_frame(b'\x00\x01\x02\x03\x04\x05\x06') is None
    # 校验和错误
    bad = bytes([FRAME_HEAD, 1, 0xF2, 0x00, 0xFF, FRAME_TAIL])
    parsed = parse_frame(bad)
    assert parsed is not None
    assert parsed.valid is False
    print('[OK] 无效帧处理测试通过')


def test_empty_payload():
    """空载荷帧"""
    frame = build_frame(addr=1, func=0xFA, payload=b'')
    # head(1) + addr(1) + func(1) + payload(0) + cs(1) + tail(1) = 5
    assert len(frame) == 5
    parsed = parse_frame(frame)
    # 命令帧被当应答帧解析时, byte[3]是cs(非错误码), 所以valid会是False
    # 这里只验证帧结构不崩
    print('[OK] 空载荷帧测试通过')


if __name__ == '__main__':
    test_build_frame_structure()
    test_checksum()
    test_parse_roundtrip()
    test_parse_response_with_error_code()
    test_parse_invalid_frame()
    test_empty_payload()
    print('\n全部协议测试通过!')
