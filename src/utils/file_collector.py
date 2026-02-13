#!/usr/bin/env python
"""File collection utilities for pattern-based file gathering"""
import shutil
from pathlib import Path
from typing import List, Set


class FileCollector:
    """文件收集器，支持排除模式和路径复制"""
    
    def __init__(self, root: Path = None):
        """初始化文件收集器
        
        Args:
            root: 根目录，默认为当前工作目录
        """
        self.root = root or Path.cwd()
    
    def get_all_files(self, exclude_dirs: Set[str] = None) -> Set[Path]:
        """获取根目录下所有文件
        
        Args:
            exclude_dirs: 要排除的目录名称集合（如 {'.git', '__pycache__', 'node_modules'}）
            
        Returns:
            所有文件的集合
        """
        if exclude_dirs is None:
            exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache'}
        
        all_files = set()
        for path in self.root.rglob('*'):
            # 检查是否在排除的目录中
            if any(excluded in path.parts for excluded in exclude_dirs):
                continue
            if path.is_file():
                all_files.add(path)
        return all_files
    
    def parse_exclude_patterns(self, patterns_file: Path) -> List[str]:
        """解析排除模式文件（类似.gitignore格式）
        
        Args:
            patterns_file: 模式文件路径
            
        Returns:
            排除模式列表
        """
        if not patterns_file.exists():
            return []
        
        patterns = []
        with open(patterns_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
        return patterns
    
    def exclude_by_patterns(self, files: Set[Path], patterns: List[str]) -> Set[Path]:
        """根据排除模式过滤文件
        
        Args:
            files: 文件集合
            patterns: glob排除模式列表
            
        Returns:
            排除后的文件集合
        """
        excluded = set()
        for pattern in patterns:
            for path in self.root.glob(pattern):
                if path.is_file() and path in files:
                    excluded.add(path)
        return files - excluded
    
    def collect_by_paths(self, file_paths: List[str]) -> Set[Path]:
        """根据路径列表收集文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            存在的文件集合
        """
        collected = set()
        for file_path in file_paths:
            file = self.root / file_path
            if file.exists() and file.is_file():
                collected.add(file)
        return collected
    
    def copy_files(self, files: Set[Path], target_dir: Path) -> None:
        """复制文件到目标目录，保持相对路径结构
        
        Args:
            files: 文件路径集合
            target_dir: 目标目录
        """
        for file in files:
            try:
                rel_path = file.relative_to(self.root)
            except ValueError:
                rel_path = file.name
            
            target_file = target_dir / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target_file)
    
    def collect_files(self, exclude_patterns: List[str] = [], required_paths: List[str] = []) -> Set[Path]:
        """收集文件：获取所有文件 -> 排除不需要的 -> 添加必需的
        
        Args:
            exclude_patterns: 要排除的glob模式列表
            required_paths: 必须包含的文件路径列表
            
        Returns:
            最终的文件集合
        """
        files = self.get_all_files()
        files = self.exclude_by_patterns(files, exclude_patterns)
        files.update(self.collect_by_paths(required_paths))
        
        return files
