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
        """根据排除模式过滤文件（支持.gitignore格式）
        
        Args:
            files: 文件集合
            patterns: gitignore格式的排除模式列表
                - /dataset: 表示根目录下的dataset目录
                - dataset: 表示任意级别的dataset子目录
                - *.txt: 表示所有后缀为.txt的文件
            
        Returns:
            排除后的文件集合
        """
        excluded = set()
        for pattern in patterns:
            pattern = pattern.strip()
            if not pattern or pattern.startswith('#'):
                continue
            
            # 检查是否以/开头（表示相对于根目录）
            is_root_relative = pattern.startswith('/')
            if is_root_relative:
                pattern = pattern[1:]
            
            # 检查是否以/结尾（表示只匹配目录）
            is_dir_only = pattern.endswith('/')
            if is_dir_only:
                pattern = pattern[:-1]
            
            # 根据模式类型进行匹配
            if is_root_relative:
                # /dataset 表示仅匹配根目录下的dataset
                self._exclude_root_pattern(files, excluded, pattern, is_dir_only)
            else:
                # dataset 表示任意路径下的dataset
                self._exclude_recursive_pattern(files, excluded, pattern, is_dir_only)
        
        return files - excluded
    
    def _exclude_root_pattern(self, files: Set[Path], excluded: Set[Path], pattern: str, is_dir_only: bool) -> None:
        """处理根目录相对模式 (如 /dataset)
        
        Args:
            files: 所有文件集合
            excluded: 排除集合（会被修改）
            pattern: 去掉前导/的模式
            is_dir_only: 是否只匹配目录
        """
        # 直接构造路径
        target_path = self.root / pattern
        
        # 检查是否存在
        if target_path.exists():
            if target_path.is_dir():
                # 排除该目录下所有文件
                for file in files:
                    if file.is_relative_to(target_path):
                        excluded.add(file)
            elif target_path.is_file() and not is_dir_only:
                # 排除指定文件
                if target_path in files:
                    excluded.add(target_path)
        return
    
    def _exclude_recursive_pattern(self, files: Set[Path], excluded: Set[Path], pattern: str, is_dir_only: bool) -> None:
        """处理递归模式 (如 dataset 或 *.txt)
        
        Args:
            files: 所有文件集合
            excluded: 排除集合（会被修改）
            pattern: 模式字符串
            is_dir_only: 是否只匹配目录
        """
        # 使用 rglob 进行递归匹配
        for path in self.root.rglob(pattern):
            if path.is_dir():
                # 排除该目录下所有文件
                for file in files:
                    if file.is_relative_to(path):
                        excluded.add(file)
            elif path.is_file() and not is_dir_only:
                # 排除指定文件
                if path in files:
                    excluded.add(path)
        return
    
    def collect_by_patterns(self, patterns: List[str]) -> Set[Path]:
        """根据 pattern 列表收集文件（支持 glob 通配符）
        
        Args:
            patterns: 文件 pattern 列表，支持 glob 通配符（如 "*.py", "src/**/*.json"）
            
        Returns:
            匹配的文件集合
        """
        collected = set()
        for pattern in patterns:
            # 移除前导的 ./ 或 ./
            pattern = pattern.lstrip('./')
            pattern = pattern.lstrip('.\\')
            
            # 使用 glob 匹配文件
            for path in self.root.glob(pattern):
                if path.is_file():
                    collected.add(path)
            
            # 同时尝试递归匹配（如果 pattern 不包含 **）
            if '**' not in pattern:
                for path in self.root.glob(f'**/{pattern}'):
                    if path.is_file():
                        collected.add(path)
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
    
    def collect_files(self, exclude_patterns: List[str] = [], include_patterns: List[str] = []) -> Set[Path]:
        """收集文件：获取所有文件 -> 排除不需要的 -> 添加必需的
        
        Args:
            exclude_patterns: 要排除的 glob 模式列表
            include_patterns: 必须包含的文件 pattern 列表（支持 glob 通配符）
            
        Returns:
            最终的文件集合
        """
        files = self.get_all_files()
        files = self.exclude_by_patterns(files, exclude_patterns)
        files.update(self.collect_by_patterns(include_patterns))
        
        return files
