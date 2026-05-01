"""注册验证模块

优先加载编译版本(_core.pyd)，如果不存在则报错。
"""

try:
    from ._core import generate_registration_code, verify_registration_code
except ImportError:
    raise ImportError(
        "找不到编译模块 _core.pyd，请确保程序完整安装。"
    )


__all__ = ['generate_registration_code', 'verify_registration_code']
