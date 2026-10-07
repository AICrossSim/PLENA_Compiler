"""String-valued enums with the same text semantics on Python 3.10+."""

from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # Python 3.10
    class StrEnum(str, Enum):
        """The subset of stdlib StrEnum used by the Compiler's contracts.

        Enum's default __str__ would emit ``LayoutKind.AFFINE_SKEW`` into
        assembly/report fields instead of their wire value ``affine_skew``.
        Keep string formatting and auto() consistent with Python 3.11.
        """

        def __new__(cls, value):
            if not isinstance(value, str):
                raise TypeError(f"{value!r} is not a string")
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        __str__ = str.__str__
        __format__ = str.__format__

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()


__all__ = ["StrEnum"]
