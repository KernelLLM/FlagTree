import os
import subprocess


def warn_stale_ascend_npu_ir_checkout(module):
    # DownloadManager reuses dst_path as-is when it already exists, so a
    # checkout left on a different commit would silently shadow the pin
    # registered for this build.
    if not module.commit_id or not os.path.isdir(os.path.join(module.dst_path, ".git")):
        return
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=module.dst_path, capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip().startswith(module.commit_id):
        return
    print(f"[WARNING] {module.dst_path} is at {head.stdout.strip()[:8]} but this build pins "
          f"AscendNPU-IR to {module.commit_id[:8]}; remove the directory (or run "
          f"'git checkout {module.commit_id}' inside it) before rebuilding")
