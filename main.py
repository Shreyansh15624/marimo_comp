import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import scipy.stats as stats
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier, go, mo, np, stats


@app.cell
def header(mo):
    mo.md("""
 
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    # **The Dead Salmons of AI Interpretability**

    In 2009, researcher Craig Bennett placed a dead Atlantic salmon in an fMRI machine and showed it pictures of humans in social situations. When analyzing the data *without correcting for multiple comparisons*, the fMRI detected "brain activity" in the dead fish.

    If your math lacks sanity checks, it will hallucinate intelligence out of pure noise.

    > **Your Challenge, Judge:** Tweak the parameters below to force our untrained, "dead fish" AI to show statistically significant intelligence.
    """)
    return


@app.cell
def controls(mo):
    sample_size = mo.ui.slider(start=500, stop=1500, step=1, value=500, label="Data Sample Size (N)")
    feature_count = mo.ui.slider(start=0, stop=100, step=5, value=5, label="Feature Count (Dimensions)")
    method = mo.ui.dropdown(
        options=["Feature Importance (Random Forest)", "Saliency Mapping (Dummy Tensor)"],
        value= "Feature Importance (Random Forest)",
        label="Interpretability Method",
    )
    p_thresh = mo.ui.number(start=0.001, stop=0.5, step=0.001, value=0.01, label="Significance Threshold (p-value)")

    title = mo.hstack(items=[mo.md("### Researcher's Control Panel")], align="center")
    controls_1 = mo.hstack(
        items = [sample_size, feature_count],
        align="center"
        )
    controls_2 = mo.hstack(
        items=[method, p_thresh],
        align="center"
        )
    control_panel = mo.vstack(
        items=[title, controls_1, controls_2],
        align="center",
        gap=0.5,
        heights=[1, 1, 1],
    ).style(padding="1rem", border="1px solid #00FFFF", border_radius="8px")
    return control_panel, feature_count, p_thresh, sample_size


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
    best_idx = np.argmin(p_values if p_values else 0)
    best_p_val = p_values[best_idx]

    # Identifying the top 10 features to making a good dashboard
    top_10_indices = np.argsort(p_values)[:10]
    top_10_p_vals = [p_values[i] for i in top_10_indices]

    # Normalizing the correlations for the chart display (0-1)
    max_corr = max(correlations)
    important_scores = [correlations[i]/max_corr for i in top_10_indices]

    # Checking if the trap is sprung
    trap_sprung = best_p_val < alpha
    return (
        alpha,
        best_p_val,
        important_scores,
        top_10_indices,
        top_10_p_vals,
        trap_sprung,
    )


@app.cell
def visualization(
    alpha,
    go,
    important_scores,
    mo,
    top_10_indices,
    top_10_p_vals,
):
    # 4. The Core Logic
    colors = ['#00FFFF' if p < alpha else "#555555" for p in top_10_p_vals]

    fig = go.Figure(data=[
        go.Bar(
            x=[f"Feature {idx}" for idx in top_10_indices],
            y=important_scores,
            marker_color=colors,
            text=["High Correlation Detected!" if p < alpha else "" for p in top_10_p_vals],
            hoverinfo="text+x",
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="hallucinaion Bar Chart - Top 10 features",
        yaxis_title="Calculated Importance",
    )

    chart = mo.ui.plotly(fig)
    return (chart,)


@app.cell
def layout(chart, control_panel, mo):
    # Combining Controls & the Chart into two columns
    dashboard = mo.vstack(
        [chart, control_panel],
    )
    dashboard
    return


@app.cell
def status_indicator(best_p_val, mo, trap_sprung):
    if trap_sprung:
        status = mo.md(f"### ✅ **TRAP SPRUNG: Statistical Illusion Achieved!** You forced a false positive out of pure noise. (Lowest p-value: {best_p_val:.4f})")
        status = status.style(color="#00FF00", background_color="#1a331a", padding="1rem", border_radius="8px")
    else:
        status = mo.md(f"### ❌ **STATUS: NO SIGNAL DETECTED.** The model is behaving like a dead fish. (Lowest p-value: {best_p_val:.4f})")
        status = status.style(color="#FF4444", background_color="#331a1a", padding="1rem", border_radius="8px")
    return (status,)


@app.cell
def post_mortem(RandomForestClassifier, mo, status):
    # A simple dummy to prove the point
    clf = RandomForestClassifier()

    mo.md(
        f"""
        {status}

        ---
        ### Post-Mortem
        The `scikit-learn` model `RandomForestClassifier()`

        The core message: Interpretability tools applied to untrained networks can still produce highly convincing, human-readable explanations.
        """
    )

    return


if __name__ == "__main__":
    app.run()
