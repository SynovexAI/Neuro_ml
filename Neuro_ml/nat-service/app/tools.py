"""Custom NAT tools registered in-process.

NAT core has no `calculator` (it only ships in an example package), so we register
our own function group here — importing this module runs the decorators.
"""
import math
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class CalculatorToolConfig(FunctionGroupBaseConfig, name="calculator"):
    include: list[str] = Field(default_factory=lambda: ["add", "subtract", "multiply", "divide", "compare"])


@register_function_group(config_type=CalculatorToolConfig)
async def calculator(_config: CalculatorToolConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=_config)

    async def _add(numbers: list[float]) -> float:
        """Add two or more numbers together."""
        if len(numbers) < 2:
            raise ValueError("Provide two or more numbers to add.")
        return sum(numbers)

    async def _subtract(numbers: list[float]) -> float:
        """Subtract the second number from the first."""
        if len(numbers) != 2:
            raise ValueError("Provide exactly two numbers to subtract.")
        return numbers[0] - numbers[1]

    async def _multiply(numbers: list[float]) -> float:
        """Multiply two or more numbers together."""
        if len(numbers) < 2:
            raise ValueError("Provide two or more numbers to multiply.")
        return math.prod(numbers)

    async def _divide(numbers: list[float]) -> float:
        """Divide the first number by the second."""
        if len(numbers) != 2:
            raise ValueError("Provide exactly two numbers to divide.")
        if numbers[1] == 0:
            raise ValueError("Cannot divide by zero.")
        return numbers[0] / numbers[1]

    async def _compare(numbers: list[float]) -> str:
        """Compare two numbers."""
        if len(numbers) != 2:
            raise ValueError("Provide exactly two numbers to compare.")
        a, b = numbers
        return f"{a} is greater than {b}" if a > b else (f"{a} is less than {b}" if a < b else f"{a} is equal to {b}")

    group.add_function(name="add", fn=_add, description=_add.__doc__)
    group.add_function(name="subtract", fn=_subtract, description=_subtract.__doc__)
    group.add_function(name="multiply", fn=_multiply, description=_multiply.__doc__)
    group.add_function(name="divide", fn=_divide, description=_divide.__doc__)
    group.add_function(name="compare", fn=_compare, description=_compare.__doc__)
    yield group
