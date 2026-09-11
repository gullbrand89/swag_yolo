from collections import Counter
from classify_failures import read_preds, classify
rows = [classify(t, p) for t, p in read_preds("runs/<stämpel>/preds_drop_0.05.txt")]
print(Counter(d for k, d in rows if k == "äkta fel" and "bins fel" in d).most_common(8))
