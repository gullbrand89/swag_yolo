"""
Flyttar och raderar det som blev kvar efter omstruktureringen.

    python stada.py          visar vad som skulle göras, rör ingenting
    python stada.py --kor    gör det

Claude kan skriva filer till din dator men inte radera eller flytta dem, så
tools/ är redan på plats medan originalen ligger kvar. Det här skriptet tar bort
dubbletterna och arkiverar det som är parkerat.

SÄKERHET: ingen fil raderas utan att ersättaren först verifierats finnas på sin
nya plats. Saknas den hoppas raderingen över och skriptet säger till.
Ingenting raderas permanent utom __pycache__ -- allt annat flyttas till _arkiv/.
"""
import shutil
import sys
from pathlib import Path

ROT = Path(__file__).resolve().parent
PKG = ROT / "transformer_post_generator"
KOR = "--kor" in sys.argv

# filer som flyttats till tools/ -- originalet raderas när ersättaren finns
FLYTTADE = [
    (ROT / "check_drift.py", ROT / "tools" / "check_drift.py"),
    (ROT / "run_report.py", ROT / "tools" / "run_report.py"),
    (ROT / "inspect_label.py", ROT / "tools" / "inspect_label.py"),
    (PKG / "classify_errors.py", ROT / "tools" / "classify_errors.py"),
    (PKG / "show_failures.py", ROT / "tools" / "show_failures.py"),
]

# hela mappen -- varje fil har en ersättare i tools/ (tests.py finns i roten)
CLAUDE_OUT = ROT / "Claude outputs"
CO_ERSATTARE = {
    "check_dwell.py": ROT / "tools" / "check_dwell.py",
    "check_visits.py": ROT / "tools" / "check_visits.py",
    "input_fidelity.py": ROT / "tools" / "input_fidelity.py",
    "label_coverage.py": ROT / "tools" / "label_coverage.py",
    "measure_levels.py": ROT / "tools" / "measure_levels.py",
    "num_breakdown.py": ROT / "tools" / "num_breakdown.py",
    "sharpness.py": ROT / "tools" / "sharpness.py",
    "tests.py": ROT / "tests.py",
}

# parkerat, inte borttaget
ARKIV = [
    (ROT / "grpo.py", "RL -- future work"),
    (ROT / "reward.py", "RL -- future work"),
    (ROT / "dwell_emitter.py", "alternativ generator, används inte"),
    (ROT / "stagger_emitter.py", "alternativ generator, används inte"),
    (ROT / "swag.py", "skiss från log-binningsutredningen, förkastad"),
    (ROT / "xx.py", "formsond, engångsbruk"),
    (ROT / "verify_range.py", "engångskontroll"),
    (ROT / "check_metrics.py", "ersatt av run_report.py"),
    (PKG / "diag_rotation.py", "diagnostik, används inte"),
]

DOKUMENT = [(ROT / "LOSS.md", ROT / "docs" / "LOSS.md")]


def logg(gjort, text):
    print(f"  {'✓' if gjort else '·'} {text}")


def main():
    print(f"projekt: {ROT}")
    print("TORRKÖRNING -- inget ändras. Kör med --kor för att genomföra.\n"
          if not KOR else "KÖR SKARPT\n")

    fel = 0

    print("radera dubbletter (ersättaren måste finnas):")
    for gammal, ny in FLYTTADE:
        if not gammal.exists():
            logg(False, f"{gammal.relative_to(ROT)} finns inte redan")
            continue
        if not ny.exists():
            print(f"  ! HOPPAR ÖVER {gammal.relative_to(ROT)} "
                  f"-- ersättaren {ny.relative_to(ROT)} saknas")
            fel += 1
            continue
        if KOR:
            gammal.unlink()
        logg(True, f"{gammal.relative_to(ROT)}  (finns som {ny.relative_to(ROT)})")

    print("\nradera mappen 'Claude outputs':")
    if CLAUDE_OUT.exists():
        saknas = [n for n, ny in CO_ERSATTARE.items()
                  if (CLAUDE_OUT / n).exists() and not ny.exists()]
        extra = [p.name for p in CLAUDE_OUT.iterdir()
                 if p.is_file() and p.name not in CO_ERSATTARE]
        if saknas:
            print(f"  ! HOPPAR ÖVER -- saknar ersättare för {saknas}")
            fel += 1
        elif extra:
            print(f"  ! HOPPAR ÖVER -- oväntade filer som inte har någon "
                  f"ersättare: {extra}")
            print("    flytta dem själv först, så vet du att inget tappas")
            fel += 1
        else:
            if KOR:
                shutil.rmtree(CLAUDE_OUT)
            logg(True, f"'Claude outputs' ({len(CO_ERSATTARE)} filer, alla har ersättare)")
    else:
        logg(False, "'Claude outputs' finns inte redan")

    print("\nflytta till _arkiv/:")
    arkiv = ROT / "_arkiv"
    if KOR:
        arkiv.mkdir(exist_ok=True)
    for p, varfor in ARKIV:
        if not p.exists():
            logg(False, f"{p.relative_to(ROT)} finns inte redan")
            continue
        if KOR:
            shutil.move(str(p), str(arkiv / p.name))
        logg(True, f"{p.relative_to(ROT)}  -- {varfor}")

    print("\nflytta till docs/:")
    docs = ROT / "docs"
    if KOR:
        docs.mkdir(exist_ok=True)
    for src, dst in DOKUMENT:
        if not src.exists():
            logg(False, f"{src.relative_to(ROT)} finns inte redan")
            continue
        if KOR:
            shutil.move(str(src), str(dst))
        logg(True, f"{src.relative_to(ROT)} -> {dst.relative_to(ROT)}")

    print("\nrensa __pycache__ (genereras om automatiskt):")
    n = 0
    for p in ROT.rglob("__pycache__"):
        if "venv" in str(p):
            continue
        if KOR:
            shutil.rmtree(p, ignore_errors=True)
        n += 1
    logg(n > 0, f"{n} mappar")

    print()
    if fel:
        print(f"{fel} steg hoppades över. Läs raderna med ! och åtgärda innan du "
              f"kör igen.")
    elif KOR:
        print("klart. Kontrollera med: python tests.py")
    else:
        print("ser det rätt ut? kör: python stada.py --kor")


if __name__ == "__main__":
    main()
