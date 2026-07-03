from rdkit.Chem import Draw
from io import BytesIO
import base64
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


def _point_in_ellipse(x, y, cx, cy, a, b):
    """Check if (x, y) is inside the ellipse centered at (cx, cy) with semi-axes a, b."""
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0


def plot_boiled_egg(mols_df, label_col=None):
    """
    Create a BOILED-Egg diagram (Daina & Zoete, 2016) using Plotly.

    Parameters:
        mols_df (DataFrame): must contain 'WLOGP' and 'TPSA' columns, and 'SMILES'
        label_col (str|None): column with activity labels (Active/Intermediate/Inactive)

    Returns:
        tuple: (plotly Figure, n_gi, n_bbb, result_df)
    """
    import plotly.graph_objects as go

    # Ellipse parameters (SwissADME values)
    GI_CX, GI_CY, GI_A, GI_B = 2.673, 71.051, 3.695, 64.118
    BBB_CX, BBB_CY, BBB_A, BBB_B = 0.267, 29.246, 1.842, 24.993

    # Classify molecules
    df = mols_df.copy()
    df["in_GI"] = df.apply(lambda r: _point_in_ellipse(r["WLOGP"], r["TPSA"], GI_CX, GI_CY, GI_A, GI_B), axis=1)
    df["in_BBB"] = df.apply(lambda r: _point_in_ellipse(r["WLOGP"], r["TPSA"], BBB_CX, BBB_CY, BBB_A, BBB_B), axis=1)
    n_gi = int(df["in_GI"].sum())
    n_bbb = int(df["in_BBB"].sum())

    # Sample if too many molecules
    plot_df = df.sample(n=300, random_state=42) if len(df) > 500 else df

    fig = go.Figure()

    # GI absorption ellipse (white/grey)
    theta = np.linspace(0, 2 * np.pi, 200)
    gi_x = GI_CX + GI_A * np.cos(theta)
    gi_y = GI_CY + GI_B * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=gi_x, y=gi_y, fill="toself",
        fillcolor="rgba(220,220,220,0.5)",
        line=dict(color="grey", width=1),
        name="GI absorption zone",
        hoverinfo="skip",
    ))

    # BBB ellipse (yellow)
    bbb_x = BBB_CX + BBB_A * np.cos(theta)
    bbb_y = BBB_CY + BBB_B * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=bbb_x, y=bbb_y, fill="toself",
        fillcolor="rgba(255,215,0,0.5)",
        line=dict(color="goldenrod", width=1),
        name="BBB zone",
        hoverinfo="skip",
    ))

    # Scatter points
    has_labels = label_col and label_col in plot_df.columns and plot_df[label_col].notna().any()
    if has_labels:
        class_colors = {"Active": "#2ca02c", "Intermediate": "#ff7f0e", "Inactive": "#d62728"}
        for cls in ["Active", "Intermediate", "Inactive"]:
            subset = plot_df[plot_df[label_col] == cls]
            if len(subset) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=subset["WLOGP"], y=subset["TPSA"],
                mode="markers",
                marker=dict(size=8, color=class_colors[cls], opacity=0.8),
                name=cls,
                text=subset["SMILES"],
                hovertemplate="<b>%{text}</b><br>WLOGP=%{x:.2f}<br>TPSA=%{y:.1f}<extra></extra>",
            ))
        # Unlabeled molecules
        unlabeled = plot_df[plot_df[label_col].isna()]
        if len(unlabeled) > 0:
            fig.add_trace(go.Scatter(
                x=unlabeled["WLOGP"], y=unlabeled["TPSA"],
                mode="markers",
                marker=dict(size=8, color="grey", opacity=0.8),
                name="Unlabeled",
                text=unlabeled["SMILES"],
                hovertemplate="<b>%{text}</b><br>WLOGP=%{x:.2f}<br>TPSA=%{y:.1f}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df["WLOGP"], y=plot_df["TPSA"],
            mode="markers",
            marker=dict(size=8, color="grey", opacity=0.8),
            name="Molecules",
            text=plot_df["SMILES"],
            hovertemplate="<b>%{text}</b><br>WLOGP=%{x:.2f}<br>TPSA=%{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=600,
        height=500,
        xaxis=dict(title="WLOGP (Lipophilicity)", range=[-2, 7],
                   gridcolor="#f0f0f0", showgrid=True),
        yaxis=dict(title="TPSA (\u00c5\u00b2)", range=[-5, 180],
                   gridcolor="#f0f0f0", showgrid=True),
        legend=dict(bgcolor="white", bordercolor="#e0e0e0", borderwidth=1),
        title=dict(text="BOILED-Egg \u2014 GI Absorption & BBB Permeability",
                   font=dict(size=14)),
        margin=dict(l=60, r=30, t=50, b=50),
    )

    return fig, n_gi, n_bbb, df


def plot_radar_chart(descriptors_dict):
    """
    Create a SwissADME-style radar/spider chart of 6 physicochemical properties.

    Parameters:
        descriptors_dict (dict): must contain keys: MW, TPSA, LogP,
            RotatableBonds, FractionCsp3

    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go

    mw = descriptors_dict["MW"]
    tpsa = descriptors_dict["TPSA"]
    logp = descriptors_dict["LogP"]
    rotb = descriptors_dict["RotatableBonds"]
    fcsp3 = descriptors_dict["FractionCsp3"]

    # Normalize to 0-1 (SwissADME ranges)
    lipo = np.clip((logp - (-2)) / (5 - (-2)), 0, 1)      # LIPO: [-2, 5] → 0..1
    size = np.clip(mw / 500.0, 0, 1)                       # SIZE: [0, 500] → 0..1
    polar = np.clip(tpsa / 140.0, 0, 1)                    # POLAR: [0, 140] → 0..1
    insolu = np.clip((logp - (-2)) / (5 - (-2)), 0, 1)    # INSOLU: [-2, 5] → 0..1
    insatu = np.clip(1.0 - fcsp3, 0, 1)                    # INSATU: 1-Fsp3
    flex = np.clip(rotb / 9.0, 0, 1)                       # FLEX: [0, 9] → 0..1

    categories = ["LIPO", "SIZE", "POLAR", "INSOLU", "INSATU", "FLEX"]
    values = [lipo, size, polar, insolu, insatu, flex]
    # Close the polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(255, 99, 132, 0.35)",
        line=dict(color="#d62728", width=2),
        marker=dict(size=5, color="#d62728"),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                showgrid=True,
                gridcolor="#e0e0e0",
                showline=False,
                dtick=0.2,
            ),
            angularaxis=dict(
                showgrid=True,
                gridcolor="#e0e0e0",
                tickfont=dict(size=15, color="#333", family="Arial Black, Arial, sans-serif"),
            ),
        ),
        showlegend=False,
        margin=dict(l=130, r=130, t=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=520,
        height=520,
    )

    return fig
