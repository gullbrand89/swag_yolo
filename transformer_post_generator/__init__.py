"""
transformer_post_generator

cfg.emitter är ett RELATIVT modulnamn (".all_emitters"). Ett relativt namn kan bara
lösas mot ett paket, och paketet är alltid det här -- inte det som råkar importera.
Därför bor upplösningen här och inte i varje verktyg. Ett verktyg som flyttas till
tools/ eller körs från en annan katalog behöver då inte veta någonting.

    from transformer_post_generator import emitter_module
    gen = emitter_module()
"""


def emitter_module():
    """-> modulen cfg.emitter pekar på, oavsett varifrån anropet kommer."""
    import importlib
    from .config import cfg
    return importlib.import_module(cfg.emitter, package=__name__)
