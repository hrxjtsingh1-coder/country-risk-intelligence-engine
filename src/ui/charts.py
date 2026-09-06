"""
Chart components for the Country Risk Intelligence Engine.
Disables zoom interactions as required by UX specifications.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_risk_score_chart(
    data: pd.DataFrame,
    country: str,
    year: int,
    show_trajectory: bool = True,
    show_peers: bool = True
) -> go.Figure:
    """
    Create a risk score trajectory chart without zoom interactions.
    
    Args:
        data: DataFrame with columns: country_iso3, year, risk_score
        country: Country ISO3 code to focus on
        year: Current year to highlight
        show_trajectory: Whether to show historical trajectory
        show_peers: Whether to show peer comparison
        
    Returns:
        Plotly figure with zoom disabled
    """
    # Filter data for the selected country
    country_data = data[data["country_iso3"] == country]
    
    if country_data.empty:
        # Create placeholder figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this country",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Risk Score",
            template="plotly_dark"
        )
        return fig
    
    # Create line chart
    fig = px.line(
        country_data,
        x="year",
        y="risk_score",
        title=f"Country Risk Score: {country}",
        labels={
            "year": "Year",
            "risk_score": "Risk Score (0-100)"
        },
        color_discrete_sequence=["#31688e"]
    )
    
    # Highlight current year
    current_data = country_data[county_data["year"] == year]
    if not current_data.empty:
        fig.add_trace(
            go.Scatter(
                x=[year],
                y=[current_data["risk_score"].values[0]] if not current_data.empty else [None],
                mode="markers",
                marker=dict(size=12, color="red", symbol="star"),
                name=f"Current: {year}",
                showlegend=False
            )
        )
    
    # Add trajectory if requested
    if show_trajectory and len(country_data) > 1:
        fig.add_scatter(
            x=country_data["year"],
            y=country_data["risk_score"],
            mode="lines+markers",
            line=dict(width=2),
            name="Risk Score",
            hovertemplate="Year: %{x}<br>Score: %{y:.1f}<extra></extra>"
        )
    
    # Configure layout to disable zoom
    fig.update_layout(
        xaxis=dict(
            title="Year",
            showgrid=True,
            gridcolor="#1a1a1a",
            zeroline=False
        ),
        yaxis=dict(
            title="Risk Score",
            showgrid=True,
            gridcolor="#1a1a1a",
            zeroline=False,
            range=[0, 100]
        ),
        title=dict(
            text=f"Country Risk Intelligence: {country} - {year}",
            x=0.5,
            xanchor="center",
            font=dict(size=16, color="white")
        ),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    # Disable zoom interactions
    fig.update_xaxes(rangeslider_visible=False, hoverformat="0f")
    fig.update_yaxes(hoverformat="0.1f")
    
    # Disable zoom buttons via config
    fig.update_layout(
        xaxis=dict(
            range=[min(country_data["year"]) - 2, max(country_data["year"]) + 2],
            showgrid=True
        ),
        yaxis=dict(
            range=[0, 100],
            showgrid=True
        ),
        plot_bgcolor="#0d121b",
        paper_bgcolor="#0d121b"
    )
    
    # Remove zoom-related modebar buttons
    fig.update_layout(
        dragmode=False,
        modebar_remove=[
            "zoom", "pan", "select", "lasso2d", "preview", "reset", "save",
            "edit", "spatial", "lasso", "contour", "histogram", "colorscale", "plotly"
        ]
    )
    
    # Disable interactive zoom features
    fig.update_xaxes(fixedrange=False)  # Allow panning but not zooming
    
    return fig


def create_peer_comparison_chart(
    data: pd.DataFrame,
    country: str,
    year: int,
    peer_group: str = "Global"
) -> go.Figure:
    """
    Create peer comparison chart without zoom interactions.
    
    Args:
        data: DataFrame with country data including peer group information
        country: Country to compare
        year: Current year
        peer_group: Peer group to display
        
    Returns:
        Plotly figure with zoom disabled
    """
    # Filter data
    country_data = data[data["country_iso3"] == country]
    peer_data = data[data["peer_group"] == peer_group]
    
    # Create bar chart
    fig = go.Figure()
    
    # Add country score
    if not country_data.empty and not country_data["risk_score"].isna().any():
        fig.add_trace(
            go.Bar(
                x=[country],
                y=[country_data["risk_score"].iloc[0]],
                name=f"{country} ({year})",
                marker_color="rgba(0, 150, 200, 0.8)",
                text=[f"{country_data['risk_score'].iloc[0]:.1f}"],
                textposition="outside",
                hovertemplate="Country: %{x}<br>Risk Score: %{y:.1f}<extra></extra>"
            )
        )
    
    # Add peer average
    if not peer_data.empty:
        peer_avg = peer_data["risk_score"].mean()
        fig.add_trace(
            go.Bar(
                x=["Peer Avg"],
                y=[peer_avg],
                name=f"Peer Group ({peer_group})",
                marker_color="rgba(255, 100, 0, 0.8)",
                text=[f"{peer_avg:.1f}"],
                textposition="outside",
                hovertemplate="Peer Group Avg: %{x}<br>Risk Score: %{y:.1f}<extra></extra>"
            )
        )
    
    # Configure layout
    fig.update_layout(
        title=f"Peer Comparison: {country} vs {peer_group} ({year})",
        xaxis_title="Entity",
        yaxis_title="Risk Score",
        template="plotly_dark",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    # Disable zoom interactions
    fig.update_layout(
        dragmode=False,
        modebar_remove=[
            "zoom", "pan", "select", "lasso2d", "preview", "reset", "save",
            "edit", "spatial", "lasso", "contour", "histogram", "colorscale", "plotly"
        ]
    )
    
    return fig


def create_risk_heatmap(
    data: pd.DataFrame,
    country: str,
    year: int
) -> go.Figure:
    """
    Create a risk factor heatmap showing contribution of different indicators.
    
    Args:
        data: DataFrame with risk driver data
        country: Country ISO3 code
        year: Current year
        
    Returns:
        Plotly heatmap figure with zoom disabled
    """
    # Filter for country and year
    filtered = data[
        (data["country_iso3"] == country) & 
        (data["year"] == year)
    ].copy()
    
    if filtered.empty:
        # Create placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No risk factor data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Risk Factor Contribution",
            xaxis_title="Indicator",
            yaxis_title="Contribution",
            template="plotly_dark"
        )
        return fig
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=filtered["weighted_contribution"].values.reshape(1, -1),
        x=filtered["indicator_code"].tolist(),
        y=[f"Score: {filtered['risk_score'].iloc[0]}"],
        colorscale="Viridis",
        colorbar=dict(title="Contribution")
    ))
    
    # Configure layout
    fig.update_layout(
        title=f"Risk Factor Contribution for {country} ({year})",
        xaxis_title="Indicator",
        yaxis_title="Risk Level",
        template="plotly_dark",
        height=400,
        margin=dict(l=120, r=20, t=60, b=40)
    )
    
    # Disable zoom interactions
    fig.update_layout(
        dragmode=False,
        modebar_remove=[
            "zoom", "pan", "select", "lasso2d", "preview", "reset", "save",
            "edit", "spatial", "lasso", "contour", "histogram", "colorscale", "plotly"
        ]
    )
    
    return fig