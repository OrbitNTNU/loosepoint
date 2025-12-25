"""
Array support module for loosepoint (fixedpoint) library.
This module adds functionality to initialize arrays of FixedPoint numbers
from Python lists and numpy arrays.
"""

import numpy as np
from typing import Union, List, Optional, Iterable


class FixedPointArray:
    """
    A wrapper class for arrays of FixedPoint numbers.
    
    This class provides convenient initialization and manipulation of arrays
    of FixedPoint numbers from Python lists or numpy arrays.
    """
    
    def __init__(
        self,
        init_array: Union[List, np.ndarray],
        signed: Optional[bool] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
        *,
        overflow: str = 'clamp',
        rounding: str = 'auto',
        overflow_alert: str = 'error',
        implicit_cast_alert: str = 'warning',
        mismatch_alert: str = 'warning',
        str_base: int = 16
    ):
        """
        Initialize an array of FixedPoint numbers.
        
        Parameters
        ----------
        init_array : list or np.ndarray
            Array-like object containing initial values (int, float, or FixedPoint)
        signed : bool, optional
            Signedness of all FixedPoint numbers in the array
        m : int, optional
            Number of integer bits for all FixedPoint numbers
        n : int, optional
            Number of fractional bits for all FixedPoint numbers
        overflow : str
            Overflow handling scheme ('clamp', 'wrap')
        rounding : str
            Rounding scheme ('auto', 'convergent', 'nearest', 'down', 'in', 'out', 'up')
        overflow_alert : str
            Overflow alert level ('error', 'warning', 'ignore')
        implicit_cast_alert : str
            Implicit cast alert level ('error', 'warning', 'ignore')
        mismatch_alert : str
            Mismatch alert level ('error', 'warning', 'ignore')
        str_base : int
            Base for string representation (2, 8, 10, or 16)
        """
        # Import FixedPoint here to avoid circular imports
        try:
            from fixedpoint import FixedPoint
        except ImportError:
            raise ImportError("fixedpoint library not found. Please install it first.")
        
        self.FixedPoint = FixedPoint
        
        # Store configuration
        self.config = {
            'signed': signed,
            'm': m,
            'n': n,
            'overflow': overflow,
            'rounding': rounding,
            'overflow_alert': overflow_alert,
            'implicit_cast_alert': implicit_cast_alert,
            'mismatch_alert': mismatch_alert,
            'str_base': str_base
        }
        
        # Convert input to numpy array for consistent handling
        if not isinstance(init_array, np.ndarray):
            init_array = np.array(init_array)
        
        self.shape = init_array.shape
        self.ndim = init_array.ndim
        self.size = init_array.size
        
        # Create flat list of FixedPoint numbers
        flat_values = init_array.flatten()
        self._fixed_points = []
        
        for value in flat_values:
            # Create FixedPoint with specified configuration
            fp = self.FixedPoint(
                value,
                signed=signed,
                m=m,
                n=n,
                overflow=overflow,
                rounding=rounding,
                overflow_alert=overflow_alert,
                implicit_cast_alert=implicit_cast_alert,
                mismatch_alert=mismatch_alert,
                str_base=str_base
            )
            self._fixed_points.append(fp)
    
    def __getitem__(self, index):
        """Get FixedPoint element(s) at the given index."""
        if isinstance(index, tuple):
            flat_index = np.ravel_multi_index(index, self.shape)
            return self._fixed_points[flat_index]
        elif isinstance(index, (int, np.integer)):
            if index < 0:
                index += self.shape[0]
            if index < 0 or index >= self.shape[0]:
                raise IndexError("FixedPointArray index out of range")

            if self.ndim == 1:
                return self._fixed_points[index]
            else:
                row_size = int(np.prod(self.shape[1:]))
                start_idx = index * row_size
                end_idx = start_idx + row_size
                sub_fps = self._fixed_points[start_idx:end_idx]
                return FixedPointArray._from_fixed_points(
                    sub_fps,
                    self.shape[1:],
                    self.config
                )
        elif isinstance(index, slice):
            sub_fps = self._fixed_points[index]
            return FixedPointArray._from_fixed_points(
                sub_fps,
                (len(sub_fps),) if self.ndim == 1 else None,
                self.config
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __iter__(self):
        if self.ndim == 1:
            # Don't use return - iterate directly
            for fp in self._fixed_points:
                yield fp
        else:
            row_size = int(np.prod(self.shape[1:]))
            for i in range(self.shape[0]):
                start = i * row_size
                end = start + row_size
                yield FixedPointArray._from_fixed_points(
                    self._fixed_points[start:end],
                    self.shape[1:],
                    self.config
                )
    
    def __setitem__(self, index, value):
        """Set FixedPoint element at the given index."""
        if isinstance(index, tuple):
            flat_index = np.ravel_multi_index(index, self.shape)
            if not isinstance(value, self.FixedPoint):
                value = self.FixedPoint(value, **self.config)
            self._fixed_points[flat_index] = value
        elif isinstance(index, (int, np.integer)):
            if not isinstance(value, self.FixedPoint):
                value = self.FixedPoint(value, **self.config)
            self._fixed_points[index] = value
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
    
    def __len__(self):
        """Return the length of the first dimension."""
        return self.shape[0] if self.shape else 0
    
    def __repr__(self):
        """String representation of the FixedPointArray."""
        return f"FixedPointArray(shape={self.shape}, qformat='{self.qformat}')"
    
    def __str__(self):
        """User-friendly string representation."""
        arr_str = self.to_numpy().__str__()
        return f"FixedPointArray({arr_str}, qformat='{self.qformat}')"
    
    @property
    def qformat(self):
        """Return the Q format of the array elements."""
        if self._fixed_points:
            return self._fixed_points[0].qformat
        return "Unknown"
    
    def to_list(self):
        """Convert to nested Python list."""
        float_array = np.array([float(fp) for fp in self._fixed_points])
        return float_array.reshape(self.shape).tolist()
    
    def to_numpy(self, dtype=None):
        """Convert to numpy array."""
        if dtype is None:
            dtype = np.float64
        
        float_array = np.array([float(fp) for fp in self._fixed_points], dtype=dtype)
        return float_array.reshape(self.shape)
    
    def to_int_array(self):
        """Convert to numpy array of integer representations."""
        int_array = np.array([int(fp) for fp in self._fixed_points], dtype=np.int64)
        return int_array.reshape(self.shape)
    
    def reshape(self, new_shape):
        """Reshape the array."""
        if np.prod(new_shape) != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}")
        
        new_array = FixedPointArray._from_fixed_points(
            self._fixed_points,
            new_shape,
            self.config
        )
        return new_array
    
    def flatten(self):
        """Return a flattened 1D FixedPointArray."""
        return self.reshape((self.size,))
    
    @classmethod
    def _from_fixed_points(cls, fixed_points, shape, config):
        """Internal method to create a FixedPointArray from existing FixedPoint objects."""
        instance = object.__new__(cls)
        instance._fixed_points = list(fixed_points)
        instance.config = config
        
        try:
            from fixedpoint import FixedPoint
            instance.FixedPoint = FixedPoint
        except ImportError:
            raise ImportError("fixedpoint library not found.")
        
        if shape is None:
            shape = (len(fixed_points),)
        
        instance.shape = shape
        instance.ndim = len(shape)
        instance.size = np.prod(shape)
        
        return instance
    
    def copy(self):
        """Create a deep copy of the FixedPointArray."""
        copied_fps = [
            self.FixedPoint(float(fp), **self.config) 
            for fp in self._fixed_points
        ]
        return FixedPointArray._from_fixed_points(
            copied_fps,
            self.shape,
            self.config.copy()
        )
    
    # Arithmetic operations
    def __add__(self, other):
        """Element-wise addition."""
        return self._elementwise_op(other, lambda a, b: a + b)
    
    def __sub__(self, other):
        """Element-wise subtraction."""
        return self._elementwise_op(other, lambda a, b: a - b)
    
    def __mul__(self, other):
        """Element-wise multiplication."""
        return self._elementwise_op(other, lambda a, b: a * b)
    
    def __truediv__(self, other):
        """Element-wise division."""
        return self._elementwise_op(other, lambda a, b: a / b)
    
    def __neg__(self):
        """Element-wise negation."""
        result_fps = [-fp for fp in self._fixed_points]
        return FixedPointArray._from_fixed_points(result_fps, self.shape, self.config)
    
    def _elementwise_op(self, other, operation):
        """Perform element-wise operation."""
        if isinstance(other, FixedPointArray):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            result_fps = [operation(a, b) for a, b in zip(self._fixed_points, other._fixed_points)]
        elif isinstance(other, (int, float, self.FixedPoint)):
            result_fps = [operation(fp, other) for fp in self._fixed_points]
        elif isinstance(other, (list, np.ndarray)):
            other_fpa = FixedPointArray(other, **self.config)
            return self._elementwise_op(other_fpa, operation)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")
        
        return FixedPointArray._from_fixed_points(result_fps, self.shape, self.config)


def from_array(
    init_array: Union[List, np.ndarray],
    signed: Optional[bool] = None,
    m: Optional[int] = None,
    n: Optional[int] = None,
    **kwargs
) -> FixedPointArray:
    """Convenience function to create a FixedPointArray."""
    return FixedPointArray(init_array, signed=signed, m=m, n=n, **kwargs)


def zeros(shape, signed=None, m=None, n=None, **kwargs):
    """Create a FixedPointArray filled with zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    init_array = np.zeros(shape)
    return FixedPointArray(init_array, signed=signed, m=m, n=n, **kwargs)


def ones(shape, signed=None, m=None, n=None, **kwargs):
    """Create a FixedPointArray filled with ones."""
    if isinstance(shape, int):
        shape = (shape,)
    init_array = np.ones(shape)
    return FixedPointArray(init_array, signed=signed, m=m, n=n, **kwargs)


def full(shape, fill_value, signed=None, m=None, n=None, **kwargs):
    """Create a FixedPointArray filled with a specific value."""
    if isinstance(shape, int):
        shape = (shape,)
    init_array = np.full(shape, fill_value)
    return FixedPointArray(init_array, signed=signed, m=m, n=n, **kwargs)
