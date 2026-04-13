from dataclasses import dataclass, field
import sympy as sym


@dataclass
class VariableGroup:
    """Holds a list of variables (e.g., 'evolution', 'constraint')."""

    name: str
    variables: list[sym.Symbol] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    name: str
    idx_str: str = "[pp]"
    groups: dict[str, VariableGroup] = field(default_factory=dict)

    def add_variable(self, group_name: str, var: sym.Symbol):
        if group_name not in self.groups:
            self.groups[group_name] = VariableGroup(name=group_name)
        self.groups[group_name].variables.append(var)

    @property
    def all_symbols(self) -> list[sym.Symbol]:
        """Helper to get every symbol in the project for differentiation checks."""
        symbols = []
        for group in self.groups.values():
            symbols.extend(group.variables)
        return symbols
