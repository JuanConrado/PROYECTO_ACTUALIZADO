"""
Estilo de visualización unificado.
Importar set_style() al inicio de cada notebook.
"""
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLT_STYLE, FIG_SIZE_TS, PALETTE


def set_style():
    """Aplica el estilo unificado de gráficos."""
    plt.style.use(PLT_STYLE)
    plt.rcParams.update({
        "figure.figsize": FIG_SIZE_TS,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    sns.set_palette("muted")


def color_for(model_name: str) -> str:
    """Devuelve el color asignado al modelo (en lowercase, sin espacios)."""
    key = model_name.lower().replace(" ", "_").replace("-", "_")
    return PALETTE.get(key, "#666666")


def save_fig(fig, name, dpi=300):
    """Guarda figura en outputs/figures/{name}.png."""
    from src.config import FIGURES_DIR
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
