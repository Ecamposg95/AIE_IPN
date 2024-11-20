import os
from flask import Flask, render_template, request
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

app = Flask(__name__)

# Carpeta para guardar gráficos
GRAPH_FOLDER = "graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)


def bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H):
    """Calcula la probabilidad posterior usando el Teorema de Bayes."""
    prior_not_H = 1 - prior_H
    denominator = (likelihood_E_given_H * prior_H) + (likelihood_E_given_not_H * prior_not_H)
    posterior_H = (likelihood_E_given_H * prior_H) / denominator
    return posterior_H


def plot_probabilities(priors, posteriors):
    """Genera un gráfico de cómo cambian las probabilidades."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(priors))), y=priors, mode='lines+markers', name='Prior (P(H))'))
    fig.add_trace(go.Scatter(x=list(range(len(posteriors))), y=posteriors, mode='lines+markers', name='Posterior (P(H|E))'))

    fig.update_layout(title="Cambio de Probabilidades",
                      xaxis_title="Iteraciones",
                      yaxis_title="Probabilidad",
                      template="plotly_white")
    graph_path = os.path.join(GRAPH_FOLDER, "probabilities.html")
    fig.write_html(graph_path)
    return graph_path


def create_bayesian_network():
    """Genera un gráfico de la red bayesiana."""
    G = nx.DiGraph()
    G.add_edges_from([('Hipótesis (H)', 'Evidencia (E1)'), ('Hipótesis (H)', 'Evidencia (E2)')])

    # Dibujar la red bayesiana
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_color="black", font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={('Hipótesis (H)', 'Evidencia (E1)'): 'P(E1|H)',
                                                      ('Hipótesis (H)', 'Evidencia (E2)'): 'P(E2|H)'})
    graph_path = os.path.join(GRAPH_FOLDER, "bayesian_network.png")
    plt.savefig(graph_path)
    plt.close()
    return graph_path


@app.route("/", methods=["GET", "POST"])
def bayes_form():
    priors = []
    posteriors = []

    if request.method == "POST":
        try:
            # Capturar los valores del formulario
            prior_H = float(request.form["prior_H"])
            likelihood_E_given_H = float(request.form["likelihood_E_given_H"])
            likelihood_E_given_not_H = float(request.form["likelihood_E_given_not_H"])

            # Validar las probabilidades
            if not (0 <= prior_H <= 1 and 0 <= likelihood_E_given_H <= 1 and 0 <= likelihood_E_given_not_H <= 1):
                raise ValueError("Todas las probabilidades deben estar entre 0 y 1.")

            # Primera iteración
            priors.append(prior_H)
            posterior_H = bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H)
            posteriors.append(posterior_H)

            # Comprobar si hay evidencia adicional
            if "extra_likelihood_E_given_H" in request.form and "extra_likelihood_E_given_not_H" in request.form:
                extra_likelihood_E_given_H = request.form["extra_likelihood_E_given_H"]
                extra_likelihood_E_given_not_H = request.form["extra_likelihood_E_given_not_H"]

                if extra_likelihood_E_given_H and extra_likelihood_E_given_not_H:
                    extra_likelihood_E_given_H = float(extra_likelihood_E_given_H)
                    extra_likelihood_E_given_not_H = float(extra_likelihood_E_given_not_H)

                    if not (0 <= extra_likelihood_E_given_H <= 1 and 0 <= extra_likelihood_E_given_not_H <= 1):
                        raise ValueError("Probabilidades adicionales deben estar entre 0 y 1.")

                    # Actualizar posterior con nueva evidencia
                    posterior_H = bayes_update(posterior_H, extra_likelihood_E_given_H, extra_likelihood_E_given_not_H)
                    priors.append(priors[-1])
                    posteriors.append(posterior_H)

            # Generar gráficos
            probabilities_graph = plot_probabilities(priors, posteriors)
            network_graph = create_bayesian_network()

            # Renderizar con gráficos
            return render_template("form.html", posterior=posterior_H, error=None, probabilities_graph=probabilities_graph, network_graph=network_graph)
        except ValueError as e:
            return render_template("form.html", posterior=None, error=str(e))

    return render_template("form.html", posterior=None, error=None, probabilities_graph=None, network_graph=None)


if __name__ == "__main__":
    app.run(debug=True)
