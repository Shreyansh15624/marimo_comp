import marimo

__generated_with = "0.23.1"
app = marimo.App(
    width="full",
    app_title="AI Interpretability: The Dead Salmon",
    auto_download=["html", "ipynb"],
)


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import scipy.stats as stats
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestClassifier

    return go, mo, np, stats


@app.cell
def header(mo):
    mo.md("""
    # **The Dead Salmons of AI Interpretability**

    In 2009, researcher Craig Bennett placed a dead Atlantic salmon in an fMRI machine and showed it pictures of humans in social situations. When analyzing the data *without correcting for multiple comparisons*, the fMRI detected "brain activity" in the dead fish.

    If your math lacks sanity checks, it will hallucinate intelligence out of pure noise.

    > **Your Challenge, Judge:** Tweak the parameters below to force our untrained, "dead fish" AI to show statistically significant intelligence.
    """)
    return


@app.cell
def controls(mo):
    sample_size = mo.ui.slider(start=100, stop=5000, step=1, value=100, label="Data Sample Size (N)")
    feature_count = mo.ui.slider(start=100, stop=5000, step=1, value=100, label="Feature Count (Dimensions)")
    method = mo.ui.dropdown(
        options=["Feature Importance (Random Forest)", "Saliency Mapping (Dummy Tensor)"],
        value= "Feature Importance (Random Forest)",
        label="Interpretability Method",
    )
    p_thresh = mo.ui.number(start=0.001, stop=0.5, step=0.001, value=0.01, label="Significance Threshold (p-value)")

    title = mo.hstack(items=[mo.md("### Researcher's Control Panel")], align="center")
    controls = mo.vstack(
        items = [sample_size, feature_count, method, p_thresh],
        align="stretch"
    )
    control_panel = mo.vstack(
        items=[title, controls],
        align="center",
        gap=0.5,
        heights=[1, 1, 1],
    ).style(padding="1rem", border="1px solid #00FFFF", border_radius="8px")
    return control_panel, feature_count, method, p_thresh, sample_size


@app.cell
def engine(feature_count, np, p_thresh, sample_size, stats):
    # 1. To Generate Pure Noise
    N = sample_size.value
    m = feature_count.value
    alpha = p_thresh.value

    Y = np.random.randn(N)
    X = np.random.randn(N, m)

    # 2. Hunting down the Fluke!
    correlations = []
    p_values = []

    for i in range(m):
        r, p = stats.pearsonr(X[:, i], Y)
        correlations.append(abs(r))
        p_values.append(p)

    # 3. Singling out the best of False Positives
    best_idx = np.argmin(p_values if p_values else [0])
    best_p_val = p_values[best_idx]

    # Identifying the top 10 features to making a good dashboard
    top_10_indices = np.argsort(p_values)[:10]
    top_10_p_vals = [p_values[i] for i in top_10_indices]

    # Normalizing the correlations for the chart display (0-1)
    max_corr = max(correlations)
    important_scores = [correlations[i]/max_corr for i in top_10_indices]

    # Checking if the trap is sprung
    trap_sprung = best_p_val < alpha
    return alpha, best_p_val, m, p_values, trap_sprung


@app.cell
def visualization(alpha, go, m, method, mo, np, p_values):
    # 1. Generating stable 3D coordinates for the 'brain/tensor' structure
    # We use static seed so the cloud shape doesn't violently jump around when tweaking sliders
    np.random.seed(42)
    # Morphing the 3D shape based on the Interpretability Method
    if "Random Forest" in method.value:
        # Organic Sphercal Brain Cloud Shape (Gaussian Distribution)
        x_coords = np.random.randn(m)
        y_coords = np.random.randn(m)
        z_coords = np.random.randn(m)
        sg, zl = False, False
    else:
        # Rigid uniform tensor cube (Uniform Distribution)
        # Scaled from -3 to 3 to roughly match the visual volume of the Gaussian Cloud
        x_coords = np.random.uniform(-3, 3, m)
        y_coords = np.random.uniform(-3, 3, m)
        z_coords = np.random.uniform(-3, 3, m)
        sg, zl = True, True

    # 2. Color & Size Logic
    marker_sizes = [24 if p < alpha else 12 for p in p_values]
    marker_colors = ['rgba(255, 105, 38, 1)' if p < alpha else 'rgba(166, 68, 68, 0.2)' for p in p_values]

    # 3. Building the WebGL Scatter Plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(width=0),
        ),
        text=[f"Feature {i} | p-value: {p:.4f}" for i, p in enumerate(p_values)],
        hoverinfo="text",
    )])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(showgrid=sg, zeroline=zl, showticklabels=False, title='', showbackground=False),
            yaxis=dict(showgrid=sg, zeroline=zl, showticklabels=False, title='', showbackground=False),
            zaxis=dict(showgrid=sg, zeroline=zl, showticklabels=False, title='', showbackground=False),
            bgcolor='rgba(40, 40, 40, 0.3)'
        )
    )

    chart = mo.ui.plotly(fig)
    return (chart,)


@app.cell
def layout(chart, control_panel, mo, ui_card):
    # Using strict CSS Grid to force Plotly to respect the column boundaries
    dashboard = mo.md(
        f"""
        <div style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 2rem; align-items: stretch; max-width: 100%;">
            <div style="display: flex; flex-direction: column; gap: 1.5rem; min-width: 0;">
                {chart}
                {control_panel}
            </div>
            <div style="min-width: 0; height: 100%;">
                <div style="height: 100%;">
                    {ui_card}
                </div>
            </div>
        </div>
        """
    )
    dashboard
    return


@app.cell
def post_mortem(best_p_val, mo, trap_sprung):
    if not trap_sprung:
        content = f"""
        ### 🔬 The Experiment is Ongoing!
        The neural matrix is currently stable, so adjust the features and significance thresholds in the control panel to see if you can force the interpretability tools to hallucinate a false positive.

        **Current Lowest p-value:** `{best_p_val:.4f}`
        """
        border_color = "#00FFFF" # Calm Cyan, isn't too harsh on the eyes
    else:
        content=f"""
        ### 🚨 TRAP SPRUNG: The alphaXiv Connection
        **Lowest p-value achieved: `{best_p_val:.4f}`**

        You just successfully replicated the **Dead Salmon Effect** inside an AI model.

        **The Illusion:** Just like the infamous 2009 fMRI experiment detected "social awareness" in a dead Atlantic salmon, you just forced a standard interpretability tool to find a highly significant, critical feature inside pure, unadulterated noise. 

        **The Mechanics:** By increasing the feature dimensions, you exponentially drove up the Family-Wise Error Rate. The model didn't find intelligence; it just rolled the mathematical dice enough times to guarantee a statistical fluke, and then painted it bright orange.

        **The Real-World Stakes:** This is the core thesis of the alphaXiv research paper. When we scale architectures to billions of parameters, this Multiple Comparisons Problem scales exponentially with it. If our mathematical sanity checks fail, our interpretability tools won't fail quietly—they will confidently lie to us, projecting profound, human-readable meaning onto a completely dead matrix.
        """
        border_color = "#FF6926" # Burnt orange to match the hallucinated node

    # Wrapping it in a sleek, semi-transparent card with a dynamic left border
    ui_card = mo.md(content).style(
        padding="1.5rem",
        border_radius="8px",
        background_color="rgba(255, 255, 255, 0.03)",
        border_left=f"4px solid {border_color}",
        margin_top="2rem",
    )

    return (ui_card,)


if __name__ == "__main__":
    app.run()
