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

    # Ellipse parameters from Daina & Zoete 2016
    GI_CX, GI_CY = 2.740, 69.105
    GI_A, GI_B = 3.600, 69.105
    BBB_CX, BBB_CY = 1.466, 26.278
    BBB_A, BBB_B = 2.400, 33.000

    # Classify molecules
    df = mols_df.copy()
    df["in_GI"] = df.apply(lambda r: _point_in_ellipse(r["WLOGP"], r["TPSA"], GI_CX, GI_CY, GI_A, GI_B), axis=1)
    df["in_BBB"] = df.apply(lambda r: _point_in_ellipse(r["WLOGP"], r["TPSA"], BBB_CX, BBB_CY, BBB_A, BBB_B), axis=1)
    n_gi = int(df["in_GI"].sum())
    n_bbb = int(df["in_BBB"].sum())

    # Sample if too many molecules
    sampled = False
    if len(df) > 300:
        plot_df = df.sample(n=300, random_state=42)
        sampled = True
    else:
        plot_df = df

    fig = go.Figure()

    # Draw ellipses
    theta = np.linspace(0, 2 * np.pi, 300)

    # GI ellipse (egg white) — drawn first as background
    gi_x = 2.740 + 3.600 * np.cos(theta)
    gi_y = 69.105 + 69.105 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=gi_x.tolist(), y=gi_y.tolist(), fill='toself',
        fillcolor='rgba(230,230,230,0.6)',
        line=dict(color='#999999', width=2),
        name='GI absorption', showlegend=True, hoverinfo='skip',
    ))

    # BBB ellipse (yolk) — drawn on top
    bbb_x = 1.466 + 2.400 * np.cos(theta)
    bbb_y = 26.278 + 33.000 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=bbb_x.tolist(), y=bbb_y.tolist(), fill='toself',
        fillcolor='rgba(255,215,0,0.6)',
        line=dict(color='#DAA520', width=2),
        name='BBB permeation', showlegend=True, hoverinfo='skip',
    ))

    # Molecule dots — added after ellipses so they render on top
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
                marker=dict(size=10, color=class_colors[cls], opacity=0.85),
                name=cls,
                text=subset["SMILES"],
                hovertemplate="<b>%{text}</b><br>WLOGP=%{x:.2f}<br>TPSA=%{y:.1f}<extra></extra>",
            ))
        unlabeled = plot_df[plot_df[label_col].isna()]
        if len(unlabeled) > 0:
            fig.add_trace(go.Scatter(
                x=unlabeled["WLOGP"], y=unlabeled["TPSA"],
                mode="markers",
                marker=dict(size=10, color="#555555", opacity=0.85),
                name="Unlabeled",
                text=unlabeled["SMILES"],
                hovertemplate="<b>%{text}</b><br>WLOGP=%{x:.2f}<br>TPSA=%{y:.1f}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df["WLOGP"], y=plot_df["TPSA"],
            mode="markers",
            marker=dict(size=10, color="#555555", opacity=0.85),
            name="Molecules",
            text=plot_df["SMILES"],
            hovertemplate="<b>%{text}</b><br>WLOGP=%{x:.2f}<br>TPSA=%{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=700,
        height=550,
        xaxis=dict(
            title=dict(text='WLOGP (Lipophilicity)', font=dict(size=14)),
            range=[-3, 7.5],
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            showgrid=True,
            dtick=1,
        ),
        yaxis=dict(
            title=dict(text='TPSA (\u00c5\u00b2)', font=dict(size=14)),
            range=[-10, 200],
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            showgrid=True,
            dtick=20,
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc',
            borderwidth=1,
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=40, b=60),
        title=dict(text='BOILED-Egg \u2014 GI Absorption & BBB Permeability',
                   font=dict(size=15, color='#333')),
    )

    return fig, n_gi, n_bbb, df, sampled


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
