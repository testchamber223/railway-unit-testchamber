# Pilot: Ra = 1e5 on the 40 x 40 mesh. Fluent working directory must be
# ansys/_fluent_runs/case0; console: /file/read-journal ../../fluent_journals/case0_pilot.py
import os
exec(open(os.path.join("..", "..", "fluent_journals", "case0_fluent.py"), encoding="utf-8").read())
main(["Ra1e5"], ["coarse"])
