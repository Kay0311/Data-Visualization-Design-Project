import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("application_data_new.csv", low_memory=False)

# --------------------------------------------------
# Compute correlation-weighted external score
# --------------------------------------------------
ext_cols = ["EXT_SOURCE_2", "EXT_SOURCE_3"]

corrs = df[ext_cols + ["TARGET"]].corr()["TARGET"][ext_cols].abs()
weights = corrs / corrs.sum()

w2 = weights["EXT_SOURCE_2"]
w3 = weights["EXT_SOURCE_3"]

df["EXT_SCORE_WEIGHTED"] = (
    w2 * df["EXT_SOURCE_2"].fillna(df["EXT_SOURCE_2"].median()) +
    w3 * df["EXT_SOURCE_3"].fillna(df["EXT_SOURCE_3"].median())
)

bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

df["EXT_WEIGHTED_BAND"] = pd.cut(
    df["EXT_SCORE_WEIGHTED"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# --------------------------------------------------
# 1. TARGET DISTRIBUTION (DONUT)
# --------------------------------------------------
target_counts = df["TARGET"].value_counts().reset_index()
target_counts.columns = ["TARGET", "count"]

donut_fig = px.pie(
    target_counts,
    names="TARGET",
    values="count",
    hole=0.5,
    title="Target Distribution (Default vs Non-Default)",
    color="TARGET",
    color_discrete_map={0: "#4CAF50", 1: "#F44336"}
)
donut_fig.update_traces(textinfo="percent+label")

# --------------------------------------------------
# 2. GENDER × TARGET (STACKED BAR)
# --------------------------------------------------
gender_target = (
    df.groupby(["CODE_GENDER", "TARGET"])
    .size()
    .reset_index(name="count")
)

gender_target = gender_target[
    ~gender_target["CODE_GENDER"].isin(["XNA", "Unknown"])
]
gender_target["TARGET"] = gender_target["TARGET"].astype(str)

gender_fig = px.bar(
    gender_target,
    x="CODE_GENDER",
    y="count",
    color="TARGET",
    barmode="stack",
    title="Target Distribution by Gender (Excluding Unknown)",
    labels={
        "CODE_GENDER": "Gender",
        "count": "Number of Applicants",
        "TARGET": "Default Status"
    },
    color_discrete_map={"0": "#4CAF50", "1": "#F44336"}
)

# --------------------------------------------------
# 3. REJECTION REASONS (STACKED BAR)
# --------------------------------------------------
group_col = "AGE_BAND"

reject_cols = [
    "prev_hc_reject_count",
    "prev_verif_reject_count",
    "prev_limit_reject_count",
    "prev_client_reject_count",
    "prev_sco_scofr_reject_count",
    "prev_system_reject_count",
]

reject_agg = df.groupby(group_col)[reject_cols].sum().reset_index()

label_map = {
    "prev_hc_reject_count": "HC",
    "prev_verif_reject_count": "VERIF",
    "prev_limit_reject_count": "LIMIT",
    "prev_client_reject_count": "CLIENT",
    "prev_sco_scofr_reject_count": "SCO / SCOFR",
    "prev_system_reject_count": "SYSTEM",
}

color_map = {
    "prev_hc_reject_count": "#E53935",
    "prev_verif_reject_count": "#FB8C00",
    "prev_limit_reject_count": "#8E24AA",
    "prev_client_reject_count": "#1E88E5",
    "prev_sco_scofr_reject_count": "#6D4C41",
    "prev_system_reject_count": "#546E7A",
}

reject_fig = go.Figure()
for col in reject_cols:
    reject_fig.add_trace(
        go.Bar(
            x=reject_agg[group_col],
            y=reject_agg[col],
            name=label_map[col],
            marker_color=color_map[col]
        )
    )

reject_fig.update_layout(
    barmode="stack",
    title="Distribution of Rejection Reasons by Age Group",
    xaxis_title="Age Band",
    yaxis_title="Total Rejection Count",
)

# --------------------------------------------------
# 4. WEIGHTED EXTERNAL SCORE × DEFAULT RATE
# --------------------------------------------------
weighted_grp = (
    df.groupby("EXT_WEIGHTED_BAND")["TARGET"]
    .mean()
    .reset_index()
)

weighted_grp["default_rate_pct"] = weighted_grp["TARGET"] * 100

weighted_ext_fig = px.bar(
    weighted_grp,
    x="EXT_WEIGHTED_BAND",
    y="default_rate_pct",
    title="Default Rate vs Correlation-Weighted External Score",
    labels={
        "EXT_WEIGHTED_BAND": "Combined External Score Band",
        "default_rate_pct": "Default Rate (%)"
    },
    color_discrete_sequence=["steelblue"]
)

weighted_ext_fig.update_layout(
    yaxis=dict(range=[0, 50])
)

# --------------------------------------------------
# DASH APP
# --------------------------------------------------
app = Dash(__name__)

app.layout = html.Div([

    html.H1("Loan Repayment Risk Dashboard", style={"textAlign": "center"}),

    # Row 1
    html.Div([
        dcc.Graph(figure=donut_fig),
        dcc.Graph(figure=gender_fig)
    ], style={"display": "flex", "justifyContent": "space-around"}),

    # Row 2 – External Source 2 / 3
    html.Div([

        html.Div([
            html.Label("Select External Risk Source"),
            dcc.Dropdown(
                id="ext_source_selector",
                options=[
                    {"label": "External Source 2", "value": "EXT_SOURCE_2"},
                    {"label": "External Source 3", "value": "EXT_SOURCE_3"},
                ],
                value="EXT_SOURCE_2",
                clearable=False
            )
        ], style={"width": "40%", "margin": "20px auto"}),

        dcc.Graph(id="ext_default_rate_fig")

    ], style={"width": "90%", "margin": "auto"}),

    # Row 3 – Weighted External Score
    html.Div([
        dcc.Graph(figure=weighted_ext_fig)
    ], style={"width": "90%", "margin": "auto"}),

    # Row 4 – Rejection Reasons
    html.Div([
        dcc.Graph(figure=reject_fig)
    ], style={"width": "90%", "margin": "auto"})

])

# --------------------------------------------------
# CALLBACK: EXT_SOURCE_2 / EXT_SOURCE_3
# --------------------------------------------------
@app.callback(
    Output("ext_default_rate_fig", "figure"),
    Input("ext_source_selector", "value")
)
def update_ext_default_rate(ext_choice):

    band_map = {
        "EXT_SOURCE_2": "EXT_SOURCE_2_BAND",
        "EXT_SOURCE_3": "EXT_SOURCE_3_BAND",
    }

    band_col = band_map[ext_choice]

    grp = (
        df.groupby(band_col)["TARGET"]
        .mean()
        .reset_index()
    )

    grp["default_rate_pct"] = grp["TARGET"] * 100

    fig = px.bar(
        grp,
        x=band_col,
        y="default_rate_pct",
        title=f"Default Rate by {ext_choice}",
        labels={
            band_col: "External Score Band",
            "default_rate_pct": "Default Rate (%)"
        },
        color_discrete_sequence=["#1f77b4" if ext_choice=="EXT_SOURCE_2" else "#ff7f0e"]
    )

    fig.update_layout(
        yaxis=dict(range=[0, 30])
    )

    return fig

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
