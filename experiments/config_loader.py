# config_loader.py - 配置加载与解析
"""
加载YAML配置，展开默认值，生成config_resolved.yaml
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置，合并默认值
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        解析后的配置字典
    """
    # 加载默认配置
    with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        default_config = yaml.safe_load(f)
    
    # 如果提供了自定义配置，合并
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        config = _deep_merge(default_config, user_config)
    else:
        config = default_config
    
    return config

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def save_resolved_config(config: Dict[str, Any], output_path: str):
    """保存解析后的配置（含默认值）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
