"""
QGAI Quantum Financial Modeling - TReDS MVP
Helper Functions

This module provides common utility functions used across the system.

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import hashlib


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Example:
        >>> output_dir = ensure_directory("data/outputs")
        >>> output_dir.exists()
        True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_filename(
    base_name: str,
    extension: str = "csv",
    include_time: bool = True
) -> str:
    """
    Generate a filename with timestamp.

    Args:
        base_name: Base name for the file
        extension: File extension (without dot)
        include_time: Include time in timestamp (default: True)

    Returns:
        Filename string with timestamp

    Example:
        >>> filename = timestamp_filename("predictions", "csv")
        >>> filename  # "predictions_20260107_143022.csv"
    """
    if include_time:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
    return f"{base_name}_{timestamp}.{extension}"


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON

    Example:
        >>> config = load_json("config.json")
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> Path:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save to
        indent: JSON indentation (default: 2)
        ensure_ascii: Whether to escape non-ASCII characters

    Returns:
        Path to saved file

    Example:
        >>> save_json({"key": "value"}, "output.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
    return filepath


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        Loaded object

    Example:
        >>> model = load_pickle("model.pkl")
    """
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> Path:
    """
    Save data to a pickle file.

    Args:
        data: Object to save
        filepath: Path to save to

    Returns:
        Path to saved file

    Example:
        >>> save_pickle(model, "model.pkl")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    return filepath


def generate_id(prefix: str, index: int, width: int = 4) -> str:
    """
    Generate a formatted ID string.

    Args:
        prefix: ID prefix (e.g., "B" for buyer)
        index: Numeric index
        width: Zero-padding width

    Returns:
        Formatted ID string

    Example:
        >>> generate_id("B", 42, 4)
        'B0042'
    """
    return f"{prefix}{index:0{width}d}"


def hash_string(text: str, length: int = 8) -> str:
    """
    Generate a short hash of a string.

    Args:
        text: String to hash
        length: Length of hash to return

    Returns:
        Hexadecimal hash string

    Example:
        >>> hash_string("test")
        '9f86d081'
    """
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def format_currency(amount: float, currency: str = "INR", use_symbol: bool = True) -> str:
    """
    Format a number as currency.

    Args:
        amount: Amount to format
        currency: Currency code
        use_symbol: Use currency symbol (₹) or code (INR)

    Returns:
        Formatted currency string

    Example:
        >>> format_currency(1234567.89)
        'INR 12,34,567.89'
        >>> format_currency(1234567.89, use_symbol=True)
        '₹12,34,567.89'
    """
    if currency == "INR":
        # Indian numbering system (lakhs, crores)
        s = f"{amount:,.2f}"
        # Convert to Indian format
        parts = s.split('.')
        integer_part = parts[0].replace(',', '')
        decimal_part = parts[1] if len(parts) > 1 else "00"

        if len(integer_part) <= 3:
            formatted = integer_part
        else:
            formatted = integer_part[-3:]
            remaining = integer_part[:-3]
            while remaining:
                formatted = remaining[-2:] + "," + formatted
                remaining = remaining[:-2]
            formatted = formatted.lstrip(',')

        prefix = "₹" if use_symbol else "INR "
        return f"{prefix}{formatted}.{decimal_part}"
    else:
        return f"{currency} {amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal as percentage.

    Args:
        value: Decimal value (0-1)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Example:
        >>> format_percentage(0.7523)
        '75.23%'
    """
    return f"{value * 100:.{decimals}f}%"


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    separator: str = '_'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Key prefix for recursion
        separator: Key separator

    Returns:
        Flattened dictionary

    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}})
        {'a_b': 1, 'a_c': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if denominator is zero

    Returns:
        Division result or default

    Example:
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 2)
        5.0
    """
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value

    Example:
        >>> clamp(1.5, 0, 1)
        1.0
    """
    return max(min_val, min(max_val, value))


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        """Stop the timer and calculate elapsed time."""
        self.elapsed = (datetime.now() - self.start_time).total_seconds()

    def __str__(self) -> str:
        """String representation of elapsed time."""
        if self.elapsed is None:
            return f"{self.name}: not completed"
        return f"{self.name}: {self.elapsed:.2f} seconds"


if __name__ == "__main__":
    # Test helper functions
    print("Testing helper functions...")

    # Test ID generation
    print(f"Generated ID: {generate_id('B', 42)}")

    # Test currency formatting
    print(f"Currency: {format_currency(1234567.89)}")

    # Test percentage formatting
    print(f"Percentage: {format_percentage(0.7523)}")

    # Test timer
    import time
    with Timer("Sleep test") as t:
        time.sleep(0.1)
    print(t)
