"""Register allocation helpers for the ATen PLENA compiler path."""

class RegisterAllocator:
    """Register Allocator: Manages address registers and GP registers"""

    def __init__(self, start_gp: int = 1, start_addr: int = 0, start_fp: int = 1):
        # HW OPERAND_WIDTH = 4 bits → gp0-gp15; gp0 reserved as constant 0.
        self.gp_registers = list(range(start_gp, 16))
        self.addr_registers = list(range(start_addr, 8))
        # f0 reserved as constant 0 (writing to f0 is a no-op for V_RED_MAX/V_RED_SUM).
        self.fp_registers = list(range(start_fp, 8))
        self.used_gp = []
        self.used_addr = []
        self.used_fp = []

    def allocate_gp(self, count: int = 1) -> list[int]:
        if len(self.gp_registers) < count:
            raise RuntimeError(f"Not enough GP registers available. Need {count}, have {len(self.gp_registers)}")

        allocated = self.gp_registers[:count]
        self.gp_registers = self.gp_registers[count:]
        self.used_gp.extend(allocated)
        return allocated

    def allocate_addr(self, count: int = 1) -> list[int]:
        if len(self.addr_registers) < count:
            raise RuntimeError(f"Not enough address registers available. Need {count}, have {len(self.addr_registers)}")

        allocated = self.addr_registers[:count]
        self.addr_registers = self.addr_registers[count:]
        self.used_addr.extend(allocated)
        return allocated

    def free_gp(self, registers: list[int]):
        for reg in registers:
            if reg in self.used_gp:
                self.used_gp.remove(reg)
                self.gp_registers.append(reg)
        self.gp_registers.sort()

    def free_addr(self, registers: list[int]):
        for reg in registers:
            if reg in self.used_addr:
                self.used_addr.remove(reg)
                self.addr_registers.append(reg)
        self.addr_registers.sort()

    def allocate_fp(self, count: int = 1) -> list[int]:
        if len(self.fp_registers) < count:
            raise RuntimeError(f"Not enough FP registers available. Need {count}, have {len(self.fp_registers)}")

        # Reverse allocation: prefer high-numbered regs to avoid conflicts with legacy hardcoded forward-allocation.
        allocated = list(reversed(self.fp_registers[-count:]))
        self.fp_registers = self.fp_registers[:-count]
        self.used_fp.extend(allocated)
        return allocated

    def free_fp(self, registers: list[int]):
        for reg in registers:
            if reg in self.used_fp:
                self.used_fp.remove(reg)
                self.fp_registers.append(reg)
        # Keep sorted so allocate_fp's tail-slice continues to return descending IDs.
        self.fp_registers.sort()


