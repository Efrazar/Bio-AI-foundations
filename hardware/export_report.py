# hardware/export_report.py
import io
import sys
from rich.console import Console
from rich.terminal_theme import MONOKAI

import hardware_validator  # must be in the same directory

# record=True is all you need — no console.capture() required
console = Console(record=True, width=90)

# Step 1: Redirect plain print() output to a string buffer
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()
hardware_validator.run_system_check()
sys.stdout = old_stdout  # restore stdout immediately after

# Step 2: Feed the captured text into Rich's console
# This is what gets recorded and rendered in the SVG
console.print(buffer.getvalue())

# Step 3: Export — record=True has been tracking every console.print() call
console.save_svg("system_report.svg", title="Bio-AI Hardware Report", theme=MONOKAI)
print("✅ system_report.svg saved.")
