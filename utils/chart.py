import plotly.graph_objects as go

def plot_recovery_chart(pair, entry_price, current_price, target_entry):
    """Membuat visualisasi chart simulasi recovery"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=["Entry Awal"], y=[entry_price], mode="markers+text",
                             name="Entry Awal", text=[f"{entry_price}"], textposition="top center"))
    fig.add_trace(go.Scatter(x=["Harga Sekarang"], y=[current_price], mode="markers+text",
                             name="Harga Sekarang", text=[f"{current_price}"], textposition="top center"))
    fig.add_trace(go.Scatter(x=["Target Entry"], y=[target_entry], mode="markers+text",
                             name="Target Entry", text=[f"{target_entry}"], textposition="top center"))

    fig.update_layout(
        yaxis_title="Harga (USDT)",
        title=f"Visualisasi Titik Entry vs Target Recovery ({pair})",
        template="plotly_dark",
        height=400
    )
    return fig
