"""
Städar bort det som blev kvar i roten efter omstruktureringen.

    python stada.py          visar vad som skulle göras, rör ingenting
    python stada.py --kor    gör det

Bakgrunden: koden flyttades från en platt rot in i transformer_post_generator/
(biblioteket) och tools/ (diagnostiken). Kopiorna i roten blev kvar. De importeras
inte av någonting -- de är helt enkelt de gamla filerna, med `from config import cfg`
i stället för `from transformer_post_generator.config import cfg`.

SÄKERHET
--------
Ingen fil raderas på förtroende. För varje dubblett läses både den gamla och den
nya filen, importraderna stryks, och varje kvarvarande rad i den GAMLA filen måste
återfinnas i den NYA. Saknas en enda rad hoppas raderingen över och skriptet säger
vilken rad det gäller -- då har du en ändring i roten som aldrig följde med över.

Tre filer i roten är inte kopior utan tidigare VERSIONER -- de ska skilja sig från
sina ersättare, så radvis jämförelse ger bara falsklarm på dem. De kontrolleras i
stället på att den nya funktionen faktiskt finns på plats i ersättaren. Skillnaderna
står utskrivna vid varje fil, så du ser vad du raderar.

_arkiv/ rörs inte. Körningsmapparna raderas inte utan flyttas till cfg.run_root.
"""
import re
import shutil
import sys
from pathlib import Path

ROT = Path(__file__).resolve().parent
PKG = ROT / "transformer_post_generator"
TOOLS = ROT / "tools"
KOR = "--kor" in sys.argv

# (gammal i roten, ersättare) -- ersättaren måste innehålla allt den gamla har
DUBBLETTER = [
    (ROT / "check_drift.py",   TOOLS / "check_drift.py"),
    (ROT / "facit.py",         TOOLS / "facit.py"),
    (ROT / "fotbehandling.py", TOOLS / "forbehandling.py"),
    (ROT / "num_breakdowm.py", TOOLS / "num_breakdown.py"),   # stavfel i originalet
    (ROT / "sharp.py",         TOOLS / "sharpness.py"),
    (ROT / "labels_steg2.py",  PKG / "labels.py"),
]

# Tidigare versioner. Ska INTE vara radvis lika -- kontrolleras på att det som
# tillkommit i ersättaren finns där. (gammal, ny, vad som skiljer, markörer i ny)
VERSIONER = [
    (ROT / "label_steg1.py", PKG / "labels.py",
     "gemensam N-rymd för nivåer och dwelltider; labels.py delar B/D",
     [r'f"B\{', r'f"D\{']),
    (ROT / "loss_steg2.py", PKG / "loss.py",
     "saknar hjälp-lossen på encodern (aux_loss)",
     [r"\ndef aux_loss\("]),
    (ROT / "model2.py", PKG / "model.py",
     "saknar hjälp-huvudena (AuxHeads, forward(..., return_aux)); "
     "qk_norm är dessutom ett riktigt cfg-fält nu, inte getattr",
     [r"\nclass AuxHeads\(", r"return_aux", r"self\.qk_norm = cfg\.qk_norm"]),
]

# körningsresultat som hamnat i roten i stället för under run_root
KORNINGAR = ["20260917_152559", "20260917_192801"]
RUN_ROOT = ROT / "runs"


def kropp(p):
    """Filens rader utan importer, tomrader och indrag -- det som faktiskt är kod."""
    rader = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith(("from ", "import ")):
            rader.append(s)
    return rader


def logg(gjort, text):
    print(f"  {'✓' if gjort else '·'} {text}")


def rel(p):
    try:
        return p.relative_to(ROT)
    except ValueError:
        return p


def main():
    print(f"projekt: {ROT}")
    print("TORRKÖRNING -- inget ändras. Kör med --kor för att genomföra.\n"
          if not KOR else "KÖR SKARPT\n")
    fel = 0

    print("radera dubbletter i roten (ersättaren måste innehålla allt):")
    for gammal, ny in DUBBLETTER:
        if not gammal.exists():
            logg(False, f"{rel(gammal)} finns inte redan")
            continue
        if not ny.exists():
            print(f"  ! HOPPAR ÖVER {rel(gammal)} -- ersättaren {rel(ny)} saknas")
            fel += 1
            continue
        saknas = [r for r in kropp(gammal) if r not in set(kropp(ny))]
        if saknas:
            print(f"  ! HOPPAR ÖVER {rel(gammal)} -- {len(saknas)} rad(er) finns "
                  f"inte i {rel(ny)}:")
            for r in saknas[:5]:
                print(f"      {r[:96]}")
            print("    granska själv innan du raderar -- något ändrades bara i roten")
            fel += 1
            continue
        if KOR:
            gammal.unlink()
        logg(True, f"{rel(gammal)}  (ersatt av {rel(ny)})")

    print("\nradera tidigare versioner (ersättaren måste ha det som tillkommit):")
    for gammal, ny, skillnad, markorer in VERSIONER:
        if not gammal.exists():
            logg(False, f"{rel(gammal)} finns inte redan")
            continue
        if not ny.exists():
            print(f"  ! HOPPAR ÖVER {rel(gammal)} -- ersättaren {rel(ny)} saknas")
            fel += 1
            continue
        text = ny.read_text(encoding="utf-8")
        saknade = [m for m in markorer if not re.search(m, text)]
        if saknade:
            print(f"  ! HOPPAR ÖVER {rel(gammal)} -- hittar inte {saknade} i "
                  f"{rel(ny)}, så den är kanske inte nyare trots allt")
            fel += 1
            continue
        if KOR:
            gammal.unlink()
        logg(True, f"{rel(gammal)}  -> {rel(ny)}")
        print(f"      den gamla {skillnad}")

    print(f"\nflytta körningar till {rel(RUN_ROOT)}/ (raderas inte):")
    for namn in KORNINGAR:
        src = ROT / namn
        if not src.exists():
            logg(False, f"{namn} finns inte redan")
            continue
        dst = RUN_ROOT / namn
        if dst.exists():
            print(f"  ! HOPPAR ÖVER {namn} -- {rel(dst)} finns redan")
            fel += 1
            continue
        if KOR:
            RUN_ROOT.mkdir(exist_ok=True)
            shutil.move(str(src), str(dst))
        logg(True, f"{namn}/ -> {rel(dst)}/")

    print("\nrensa __pycache__ (genereras om automatiskt):")
    n = 0
    for p in ROT.rglob("__pycache__"):
        if "venv" in str(p):
            continue
        if KOR:
            shutil.rmtree(p, ignore_errors=True)
        n += 1
    logg(n > 0, f"{n} mappar")

    print("\nlämnas orört med flit:")
    for text in ("_arkiv/            -- parkerat, bl.a. grpo.py och reward.py (RL)",
                 "tests.py           -- testsviten, körs från roten",
                 "train_fixed.py     -- bisektionsverktyget, använder paketimporter",
                 "emitterbeskrivning.pptx"):
        print(f"  · {text}")

    print()
    if fel:
        print(f"{fel} steg hoppades över. Läs raderna med ! och åtgärda innan du kör igen.")
    elif KOR:
        print("klart. Kontrollera med: python tests.py")
        print("Den här filen har gjort sitt och kan raderas.")
    else:
        print("ser det rätt ut? kör: python stada.py --kor")


if __name__ == "__main__":
    main()
