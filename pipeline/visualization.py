from rdkit.Chem import Draw
from io import BytesIO
import base64
import altair as alt
import pandas as pd
import numpy as np


def mol_to_image(mol, size=(300, 300)):
    """
    Render an RDKit Mol object as a PNG image.

    Parameters:
        mol: RDKit Mol object
        size: (width, height) tuple in pixels

    Returns:
        bytes: PNG image data
    """
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mol_to_base64_png(mol, size=(300, 300)):
    """
    Render an RDKit Mol object as a base64-encoded PNG for HTML embedding.

    Parameters:
        mol: RDKit Mol object
        size: (width, height) tuple in pixels

    Returns:
        str: base64-encoded PNG data (for use in <img src="data:image/png;base64,...">)
    """
    return base64.b64encode(mol_to_image(mol, size=size)).decode("utf-8")


def _ellipse_df(cx, cy, a, b, label, n_points=200):
    """Generate points for an ellipse outline."""
    t = np.linspace(0, 2 * np.pi, n_points)
    return pd.DataFrame({
        "x": cx + a * np.cos(t),
        "y": cy + b * np.sin(t),
        "zone": label,
    })


def _point_in_ellipse(x, y, cx, cy, a, b):
    """Check if (x, y) is inside the ellipse centered at (cx, cy) with semi-axes a, b."""
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0


def plot_boiled_egg(mols_df, label_col=None):
    """
    Create a BOILED-Egg diagram (Daina & Zoete, 2016).

    Parameters:
        mols_df (DataFrame): must contain 'WLOGP' and 'TPSA' columns, and 'SMILES'
        label_col (str|None): column with activity labels (Active/Intermediate/Inactive)

    Returns:
        tuple: (altair Chart, n_gi, n_bbb)
    """
    # Ellipse parameters
    GI_CX, GI_CY, GI_A, GI_B = 2.5, 70, 3.0, 60
    BBB_CX, BBB_CY, BBB_A, BBB_B = 0.5, 45, 2.0, 30

    # Classify each molecule
    df = mols_df.copy()
    df["in_GI"] = df.apply(lambda r: _point_in_ellipse(r["WLOGP"], r["TPSA"], GI_CX, GI_CY, GI_A, GI_B), axis=1)
    df["in_BBB"] = df.apply(lambda r: _point_in_ellipse(r["WLOGP"], r["TPSA"], BBB_CX, BBB_CY, BBB_A, BBB_B), axis=1)
    n_gi = int(df["in_GI"].sum())
    n_bbb = int(df["in_BBB"].sum())

    # Ellipse outlines
    gi_ellipse = _ellipse_df(GI_CX, GI_CY, GI_A, GI_B, "GI absorption zone")
    bbb_ellipse = _ellipse_df(BBB_CX, BBB_CY, BBB_A, BBB_B, "BBB permeability zone")
    ellipses = pd.concat([gi_ellipse, bbb_ellipse], ignore_index=True)

    # Build ellipse layer
    zone_color = alt.Scale(
        domain=["GI absorption zone", "BBB permeability zone"],
        range=["#AAAAAA", "#F0C75E"],
    )
    ellipse_layer = (
        alt.Chart(ellipses)
        .mark_line(strokeWidth=2.5, strokeDash=[6, 3])
        .encode(
            x=alt.X("x:Q", title="WLOGP (Lipophilicity)"),
            y=alt.Y("y:Q", title="TPSA (\u00c5\u00b2)"),
            color=alt.Color("zone:N", scale=zone_color, legend=alt.Legend(title="Zone")),
            detail="zone:N",
        )
    )

    # Filled ellipse backgrounds
    gi_fill = (
        alt.Chart(gi_ellipse)
        .mark_area(opacity=0.08, color="#AAAAAA")
        .encode(x="x:Q", y="y:Q")
    )
    bbb_fill = (
        alt.Chart(bbb_ellipse)
        .mark_area(opacity=0.15, color="#F0C75E")
        .encode(x="x:Q", y="y:Q")
    )

    # Scatter layer
    tooltip_fields = [
        alt.Tooltip("SMILES:N"),
        alt.Tooltip("WLOGP:Q", format=".2f"),
        alt.Tooltip("TPSA:Q", format=".1f"),
    ]

    has_labels = label_col and label_col in df.columns and df[label_col].notna().any()
    if has_labels:
        tooltip_fields.append(alt.Tooltip(f"{label_col}:N"))
        classes = [c for c in ["Active", "Intermediate", "Inactive"] if c in df[label_col].values]
        class_colors = {"Active": "#2ca02c", "Intermediate": "#ff7f0e", "Inactive": "#d62728"}
        scatter_color = alt.Color(
            f"{label_col}:N",
            scale=alt.Scale(domain=classes, range=[class_colors[c] for c in classes]),
            legend=alt.Legend(title="Activity"),
        )
    else:
        df["_class"] = "Molecule"
        scatter_color = alt.Color(
            "_class:N",
            scale=alt.Scale(domain=["Molecule"], range=["#4C72B0"]),
            legend=alt.Legend(title=""),
        )

    scatter_layer = (
        alt.Chart(df)
        .mark_circle(size=70, opacity=0.85)
        .encode(
            x="WLOGP:Q",
            y="TPSA:Q",
            color=scatter_color,
            tooltip=tooltip_fields,
        )
    )

    chart = (
        (gi_fill + bbb_fill + ellipse_layer + scatter_layer)
        .properties(
            title="BOILED-Egg Model \u2014 GI Absorption & BBB Permeability",
            height=450,
        )
        .configure_title(fontSize=16, anchor="start")
    )

    return chart, n_gi, n_bbb, df
