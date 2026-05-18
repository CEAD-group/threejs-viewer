"""Minimal reproduction of the strand_collapse cross-link bug in
parametric_tube.

Run: uv run python repro_min.py

Self-contained 100-spine-point bead (130x smaller than the original
ribweaver dump tube_8f5bba97). Scaled 100x from the source so it sits
comfortably in the viewport. Reproduces the closed-polygon "diamond
fan" artifact even though no travel move, no zero-width segment, and
no layer transition is in the slice -- the trigger is purely local.

Repro reduced by binary search from the dump (13165 points) down to
spine indices 5670..5770 (window centered at idx 5720, half=50).

Toggle ``strand_collapse=False`` below to confirm the artifact is
specific to the strand-collapse code path -- the base tube renders
fine on this slice without it.
"""

import base64
import time

import numpy as np

from threejs_viewer import viewer


N = 100   # spine length after slicing
_SPINE_W_H_B64 = (
    "IAa3PYAKIb7ALya7oHu2PQApIL7O2SO7IC+2PYCrH74+CBq74OG1PSAuH75m5gK7AJa1PUCwHr7U"
    "9My6AEi1PTAzHr5IInG6wEazPcDaGr4gmvs5QFCxPXCCF76g2fM5gGetPRDNEL6gzvQ5QJClPUhp"
    "A77Ap/Q5AOqVPbAt0b2gpvQ5gEttPUCgS72ghfQ5QOpqPYBoRr0Ag/Q5QMNlPaD1Q71AlPQ5AEwj"
    "O8BAA71gnPQ5ACLHvADjxbzgHvQ5YHMavcCxpbwgnfY5oOQ1vUCFlbwgue85gJ1DvUB+jbzA/v05"
    "YFxRvUCUhbxAnvO5gDBTvYCLhLy4uHi64ANVvcB5g7xwfLC6wNZWvQBogrw8ete6IKxYvUBjgbyc"
    "5ue6IHhbvYAdf7ywr+m6QEJevYBOe7ywuem6QKdjvYD/cbyMsem6AH1ovYCQY7yQxem6wEBrvYBC"
    "TrwYuum6QLBrvQBiOrzkyum6oJ9rvYBlMLxAium64DNrvQB+Jrw4Sd+6oOxqvYD8HrxASMK6oKRq"
    "vYBxF7y0GJq6oDZqvYATELxodUC6wMJpvQCrCLzAire5wI9kvQD6e7sA7f05oDxfvQDgKzrgse05"
    "YGRUvYCBHTwAAvc5IAM/vQAR4jzA8vQ5wHIUvVBCgj2AB/U5IKERvbCGhj3gnvU5AN0HvfAYiz2g"
    "mfQ5wKT9vACajD0AdvQ5AKSsPBCeqz2gpPQ5gNFAPSAruz1gsfQ5gPl1PXD/wj3AqfQ5IE6IPQDh"
    "xj3ArvQ54PGOPTDXyD1gq/Q5QKaVPZCZyj0ArfQ5YJuWPTDLyj1ArPQ54JGXPZD5yj3grPQ5YIiY"
    "PZAlyz2grPQ5wIGZPWA4yz3ArPQ54LmaPTBayz3ArPQ5IPCbPXBeyz3ArPQ5AF+ePTAeyz3ArPQ5"
    "ALigPeAWyj3ArPQ5QKmhPWAxyT3ArPQ5AD2iPTAPyD3ArPQ5IJaiPeAhxz3ArPQ5IKqiPRAtxj3A"
    "rPQ5QLuiPYAxxT3ArPQ5QKeiPQA4xD3ArPQ54EKgPaCiuj3ArPQ5QHadPaAfsT3ArPQ5QOmXPVAf"
    "nj3ArPQ5gMSMPYApcD3ArPQ5wPNsPUDcrzzArPQ5gIRqPYBwpTzArPQ5AE9lPQCboDzArPQ5AGgw"
    "OwDrgjvArPQ5wCPEvADOebvArPQ5oJEYvQDJ/LvArPQ5oNgzvYBnHrzArPQ5IHhBvQBXLrzArPQ5"
    "IDVPvQDVPLzArPQ5gCdRvQCRPrzArPQ5gB5TvQD7P7zArPQ5QBZVvQBPQbzArPQ5gA5XvYCgQrzA"
    "rPQ5ILdZvYBwRLzArPQ5IGZcvYDDRbzArPQ5AL5hvYCaSLzArPQ5YORnvYCzTbzArPQ5ABptvYAf"
    "W7zArPQ54ORvvQC/a7zArPQ5oPZxvQDwfbzArPQ5YKZyvYC9grzArPQ5gGpzvUBohrzArPQ54AF0"
    "vQBCirygrPQ5AKN0vcACjrzArPQ5oKp4vUAfqbyArPQ5IMV8vcDlw7xArfQ50D+CvUBw+rwArPQ5"
    "4FGGvaAMGL2ArfQ5IBeLvQByMb3AsfQ5UO6MvcBiM72gsfQ5IJSPveB4Mr0AmvQ5kAKSvaAfK72A"
    "lvQ5XPyCPHJTgjzIgH087S9xPBLVYTwFQ1I8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PH9q"
    "PDx/ajw8f2o8PH9qPDx/ajw8f2o8POS0ZTyNUXk8fGaFPODAizxNgI486sqOPOrKjjzqyo486sqO"
    "POrKjjzqyo486sqOPPgLjTzhYog8slyBPF7mcTxkR2A8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8"
    "f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PJIyXDzRcXA8WfiBPJKdiTxu54086sqOPOrKjjzq"
    "yo486sqOPOrKjjz6uY48uNaLPGpdhTxEnng8VjdkPH9qPDx/ajw8f2o8PH9qPDx/ajw8f2o8PH9q"
    "PDx/ajw8f2o8PH9qPDx/ajw8f2o8PJO+WzxodXA8MC6CPCDbiTyxCY486sqOPOrKjjzqyo486sqO"
    "POrKjjzqyo48WnyNPGhaiTw9iYI8JD10PIb3YTx/ajw8f2o8PH9qPDx/ajw8f2o8PH9qPDx/ajw8"
    "f2o8PKabRDsmX0Y7A+1PO61iYDsZ6HQ7ptuEO7x0kzuMZ5M7vHSTO2x0kztadJM7SnKTOyBykzs0"
    "c5M7t3OTO+Brkzu8dJM7giWTO7x0kzu3QGg7tCVKO0L2LzvFlhw7MFcUO7x0Ezu8dBM7vHQTOxpx"
    "Ezu8dBM7cW4TO7x0EztFrxg7TOYmO4ZHOzvTv1U7Tn1wO7x0kzsPBZM7vHSTO7x0kzu8dJM7vHST"
    "O4pzkztQcZM7O3STO7x0kzuNdJM7vHSTO6V0kzvXrnY7LfVXO25fOjsyKyM7JicWO7x0Ezu8dBM7"
    "vHQTO7x0Ezu8dBM7IqgTO9FrHDvJETA72Y1LO7yDaju8dJM7vHSTO7x0kzu8dJM7vHSTO7x0kzu8"
    "dJM7vHSTO7x0kzu8dJM7vHSTO7x0kzveXnc7uu9XOwW8OTtgcCI7Kr8VO7x0Ezu8dBM7vHQTO7x0"
    "Ezu8dBM7vHQTOyRsFzsM9yM7sKc4OywzUjuJ7W07t3STO7x0kzuxdJM7vHSTO7x0kzu8dJM7knOT"
    "O1hzkzs="
)

_buf = np.frombuffer(base64.b64decode(_SPINE_W_H_B64), dtype=np.float32)
spine = _buf[: N * 3].reshape(N, 3).copy()
widths = _buf[N * 3 : N * 3 + N].copy()
heights = _buf[N * 3 + N :].copy()

# Scale to a viewer-friendly size (geometry ratios preserved).
SCALE = 100.0
spine *= SCALE
widths *= SCALE
heights *= SCALE

print(f"spine bbox: x[{spine[:,0].min():.2f},{spine[:,0].max():.2f}] "
      f"y[{spine[:,1].min():.2f},{spine[:,1].max():.2f}] "
      f"z[{spine[:,2].min():.3f},{spine[:,2].max():.3f}]")
print(f"width range: [{widths.min():.3f}, {widths.max():.3f}]  "
      f"zero-width points: {int((widths==0).sum())}")

v = viewer()
v.clear()
v.add_parametric_tube(
    "bead",
    spine=spine,
    widths=widths,
    heights=heights,
    color=0x7AB8CC,
    opacity=1.0,
    metalness=0.15,
    roughness=0.4,
    anchor="top",
    lod=False,
    strand_collapse=True,
)

print("\nViewer up. Press 'M' for wireframe to make the diamond fan obvious.")
while True:
    time.sleep(1.0)
