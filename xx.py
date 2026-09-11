from tests import gen
from verify import verify_label

seqs, lab = gen(3, 2)[0]
print("label:", {k: v for k, v in lab.items()})
verify_label(seqs[0], lab, verbose=True)