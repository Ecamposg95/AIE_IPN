import unittest
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from red import crear_red_bayesiana, guardar_red_bayesiana, cargar_red_bayesiana

class TestRedBayesiana(unittest.TestCase):

    def setUp(self):
        # Configuración inicial de la red
        self.red = BayesianNetwork()
        self.red.add_nodes_from(["Lluvia", "Sprinkler", "Césped mojado"])
        self.red.add_edges_from([("Lluvia", "Césped mojado"), ("Sprinkler", "Césped mojado")])
        
        # Crear y agregar CPDs
        cpd_lluvia = TabularCPD(variable="Lluvia", variable_card=2, values=[[0.8], [0.2]])
        cpd_sprinkler = TabularCPD(variable="Sprinkler", variable_card=2, values=[[0.5], [0.5]])
        cpd_cesped = TabularCPD(
            variable="Césped mojado", variable_card=2,
            values=[[0.99, 0.9, 0.9, 0], [0.01, 0.1, 0.1, 1]],
            evidence=["Lluvia", "Sprinkler"],
            evidence_card=[2, 2]
        )
        self.red.add_cpds(cpd_lluvia, cpd_sprinkler, cpd_cesped)

    def test_crear_red(self):
        # Prueba para verificar que los nodos y las relaciones se agregan correctamente
        self.assertEqual(set(self.red.nodes()), {"Lluvia", "Sprinkler", "Césped mojado"})
        self.assertEqual(set(self.red.edges()), {("Lluvia", "Césped mojado"), ("Sprinkler", "Césped mojado")})

    def test_validar_modelo(self):
        # Verificar si el modelo es válido después de añadir CPDs
        self.assertTrue(self.red.check_model())

    def test_guardar_y_cargar_red(self):
        # Guardar y cargar la red en un archivo
        filename = "test_red.json"
        guardar_red_bayesiana(self.red, filename)
        red_cargada = cargar_red_bayesiana(filename)
        
        # Verificar que la red cargada tenga los mismos nodos y relaciones
        self.assertEqual(set(red_cargada.nodes()), set(self.red.nodes()))
        self.assertEqual(set(red_cargada.edges()), set(self.red.edges()))

    def test_inferir_probabilidad(self):
        # Testear la inferencia en la red con y sin evidencia
        from pgmpy.inference import VariableElimination
        infer = VariableElimination(self.red)
        
        # Inferir la probabilidad de "Césped mojado" sin evidencia
        resultado = infer.query(variables=["Césped mojado"])
        self.assertIsNotNone(resultado)

        # Inferir la probabilidad de "Césped mojado" con evidencia
        evidencia = {"Lluvia": 1, "Sprinkler": 0}
        resultado_con_evidencia = infer.query(variables=["Césped mojado"], evidence=evidencia)
        self.assertIsNotNone(resultado_con_evidencia)

if __name__ == "__main__":
    unittest.main()
