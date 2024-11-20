from flask import Flask, render_template, request

app = Flask(__name__)

def bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H):
    """Calcula la probabilidad posterior usando el Teorema de Bayes."""
    prior_not_H = 1 - prior_H
    denominator = (likelihood_E_given_H * prior_H) + (likelihood_E_given_not_H * prior_not_H)
    posterior_H = (likelihood_E_given_H * prior_H) / denominator
    return posterior_H

@app.route("/", methods=["GET", "POST"])
def bayes_form():
    iterations = []  # Lista para almacenar iteraciones
    posterior_H = None
    error = None

    if request.method == "POST":
        try:
            # Si es la primera vez, inicializamos con los datos del formulario
            if "step" not in request.form or request.form["step"] == "initial":
                prior_H = float(request.form["prior_H"])
                likelihood_E_given_H = float(request.form["likelihood_E_given_H"])
                likelihood_E_given_not_H = float(request.form["likelihood_E_given_not_H"])

                # Validar los valores iniciales
                if not (0 <= prior_H <= 1 and 0 <= likelihood_E_given_H <= 1 and 0 <= likelihood_E_given_not_H <= 1):
                    raise ValueError("Todas las probabilidades deben estar entre 0 y 1.")

                # Calcular el primer posterior
                posterior_H = bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H)
                iterations.append({
                    "evidence": 1,
                    "prior": prior_H,
                    "posterior": posterior_H
                })

                # Preparar para preguntar si se desea agregar más evidencia
                return render_template("form.html", iterations=iterations, posterior=posterior_H, step="next", error=None)

            # Si es la segunda etapa, procesamos nueva evidencia
            elif request.form["step"] == "next":
                # Recuperar la lista de iteraciones
                prior_H = float(request.form["current_posterior"])
                likelihood_E_given_H = float(request.form["extra_likelihood_E_given_H"])
                likelihood_E_given_not_H = float(request.form["extra_likelihood_E_given_not_H"])

                # Validar las probabilidades adicionales
                if not (0 <= likelihood_E_given_H <= 1 and 0 <= likelihood_E_given_not_H <= 1):
                    raise ValueError("Las probabilidades adicionales deben estar entre 0 y 1.")

                # Calcular nuevo posterior
                posterior_H = bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H)
                evidence_num = int(request.form["evidence_num"]) + 1
                iterations = eval(request.form["iterations"])  # Convertir la lista de texto a objeto Python
                iterations.append({
                    "evidence": evidence_num,
                    "prior": prior_H,
                    "posterior": posterior_H
                })

                # Preguntar nuevamente si se desea agregar más evidencia
                return render_template("form.html", iterations=iterations, posterior=posterior_H, step="next", error=None)

        except ValueError as e:
            error = str(e)

    # Renderizar el formulario inicial
    return render_template("form.html", iterations=None, posterior=None, step="initial", error=error)

if __name__ == "__main__":
    app.run(debug=True)
