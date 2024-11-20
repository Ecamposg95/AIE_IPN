def bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H):
    """
    Aplica el teorema de Bayes para calcular la probabilidad posterior.
    """
    prior_not_H = 1 - prior_H
    denominator = (likelihood_E_given_H * prior_H) + (likelihood_E_given_not_H * prior_not_H)
    posterior_H = (likelihood_E_given_H * prior_H) / denominator
    return posterior_H


def main():
    print("=== Teorema de Bayes ===\n")

    try:
        # Solicitar valores iniciales al usuario
        prior_H = float(input("Ingresa la probabilidad inicial de la hipótesis (P(H), entre 0 y 1): "))
        likelihood_E_given_H = float(input("Ingresa la probabilidad de la evidencia dado que H es cierta (P(E|H), entre 0 y 1): "))
        likelihood_E_given_not_H = float(input("Ingresa la probabilidad de la evidencia dado que H no es cierta (P(E|¬H), entre 0 y 1): "))
        
        # Validar rangos de las probabilidades
        if not (0 <= prior_H <= 1 and 0 <= likelihood_E_given_H <= 1 and 0 <= likelihood_E_given_not_H <= 1):
            raise ValueError("Todos los valores deben estar entre 0 y 1.")
        
        # Primera actualización
        posterior_H = bayes_update(prior_H, likelihood_E_given_H, likelihood_E_given_not_H)
        print(f"\nPosterior después de la primera evidencia: P(H|E) = {posterior_H:.4f}")

        # Ofrecer realizar más iteraciones
        while True:
            print("\n¿Quieres agregar nueva evidencia? (sí/no)")
            continuar = input("> ").strip().lower()
            if continuar not in ["sí", "si", "s", "yes"]:
                break

            # Solicitar nuevos valores de probabilidad condicional
            likelihood_E_given_H = float(input("Ingresa la nueva probabilidad de la evidencia dado que H es cierta (P(E|H), entre 0 y 1): "))
            likelihood_E_given_not_H = float(input("Ingresa la nueva probabilidad de la evidencia dado que H no es cierta (P(E|¬H), entre 0 y 1): "))

            # Validar rangos de las probabilidades
            if not (0 <= likelihood_E_given_H <= 1 and 0 <= likelihood_E_given_not_H <= 1):
                raise ValueError("Todos los valores deben estar entre 0 y 1.")

            # Actualizar la probabilidad posterior
            posterior_H = bayes_update(posterior_H, likelihood_E_given_H, likelihood_E_given_not_H)
            print(f"\nPosterior actualizado: P(H|E) = {posterior_H:.4f}")

        print("\n=== Resumen Final ===")
        print(f"Probabilidad final de la hipótesis (P(H|E)): {posterior_H:.4f}")

    except ValueError as e:
        print(f"Error: {e}")


# Ejecutar el programa
if __name__ == "__main__":
    main()
