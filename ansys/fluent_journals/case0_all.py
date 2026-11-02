# All Case 0 runs (Ra 1e3..1e6) on all meshes. Fluent working directory must be
# ansys/_fluent_runs/case0; console: /file/read-journal ../../fluent_journals/case0_all.py
import os
exec(open(os.path.join("..", "..", "fluent_journals", "case0_fluent.py"), encoding="utf-8").read())
main(["Ra1e3", "Ra1e4", "Ra1e5", "Ra1e6"], ["coarse", "medium", "fine"])
