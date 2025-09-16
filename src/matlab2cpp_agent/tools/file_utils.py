"""File utilities for the MATLAB to C++ conversion system."""

import shutil
from pathlib import Path
from typing import List, Optional
from loguru import logger


class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure a directory exists, creating it if necessary."""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def copy_file(source: Path, destination: Path) -> None:
        """Copy a file from source to destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    
    @staticmethod
    def find_files(directory: Path, pattern: str) -> List[Path]:
        """Find files matching a pattern in a directory."""
        return list(directory.glob(pattern))
    
    @staticmethod
    def read_file(file_path: Path, encoding: str = 'utf-8') -> str:
        """Read file content with error handling."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    @staticmethod
    def write_file(file_path: Path, content: str, encoding: str = 'utf-8') -> None:
        """Write content to a file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes."""
        return file_path.stat().st_size
    
    @staticmethod
    def is_matlab_file(file_path: Path) -> bool:
        """Check if a file is a MATLAB file."""
        return file_path.suffix.lower() in ['.m', '.mat', '.fig']
    
    @staticmethod
    def clean_directory(directory: Path) -> None:
        """Clean a directory, removing all contents."""
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


