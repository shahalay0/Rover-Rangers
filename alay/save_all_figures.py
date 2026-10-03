import contextlib
import io
import os
import runpy
import traceback
import matplotlib
matplotlib.use("Agg")                      # draw into files, no pop-up windows
import matplotlib.pyplot as plt

SCRIPTS = ["look", "compare", "evaluate", "confusion", "identify_test",
           "make_signal", "fog", "spin", "unspin", "timing", "find_start", "full_test",
           "repeat_code", "voyager", "deepspace", "demo", "demo_full"]

os.makedirs("figures", exist_ok=True)
failed = []

for name in SCRIPTS:
    count = [0]

    def save_and_close(*args, **kwargs):
        for num in plt.get_fignums():
            count[0] += 1
            plt.figure(num).savefig(f"figures/{name}_{count[0]}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    plt.show = save_and_close              # every plt.show() now saves instead of showing
    print(f"running {name}.py ...", flush=True)
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            runpy.run_path(f"{name}.py", run_name="__main__")
        save_and_close()
    except Exception:
        failed.append(name)
        printed.write("\n" + traceback.format_exc())
        plt.close("all")
    with open(f"figures/{name}.txt", "w", encoding="utf-8") as f:
        f.write(printed.getvalue())

print("\nDone! Saved to the 'figures' folder.")
print("Failed:", failed if failed else "none")