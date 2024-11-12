from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import json
import networkx as nx
import matplotlib.pyplot as plt

def crear_red_bayesiana():
    # Crear una red bayesiana vacía
    red = BayesianNetwork()
    
    # Solicitar al usuario que ingrese nodos y conexiones
    print("Defina la estructura de la red bayesiana.")
    while True:
        nodo = input("Ingrese el nombre de un nodo (o 'salir' para finalizar): ").strip()
        if nodo.lower() == 'salir':
            break
        red.add_node(nodo)

    print("Defina las relaciones entre los nodos.")
    while True:
        origen = input("Nodo de origen de la relación (o 'salir' para finalizar): ").strip()
        if origen.lower() == 'salir':
            break
        destino = input(f"Nodo de destino para la relación desde {origen}: ").strip()
        red.add_edge(origen, destino)

    # Solicitar al usuario que ingrese las tablas de probabilidades condicionales (CPDs)
    print("\nDefina las tablas de probabilidades para cada nodo.")
    cpds = []
    for nodo in red.nodes():
        padres = list(red.get_parents(nodo))
        valores = int(input(f"¿Cuántos valores tiene el nodo '{nodo}'? (por ejemplo, 2 para Sí/No): "))

        # Crear una tabla de probabilidad condicional
        if padres:
            print(f"Para el nodo '{nodo}', con padres {padres}:")
            entradas = []
            for i in range(valores ** len(padres)):
                prob = list(map(float, input(f"Probabilidad para el caso {i + 1} (en formato separado por espacios): ").split()))
                entradas.extend(prob)
            cpd = TabularCPD(variable=nodo, variable_card=valores, values=[entradas], evidence=padres, evidence_card=[valores] * len(padres))
        else:
            prob = list(map(float, input(f"Probabilidades de '{nodo}' (separadas por espacios): ").split()))
            cpd = TabularCPD(variable=nodo, variable_card=valores, values=[prob])

        cpds.append(cpd)

    # Agregar CPDs a la red
    for cpd in cpds:
        red.add_cpds(cpd)

    # Verificar si la red es válida
    if not red.check_model():
        print("La red no es válida. Verifique las probabilidades condicionales y la estructura.")
        return None
    print("Red bayesiana creada exitosamente.")
    return red

def guardar_red_bayesiana(red, filename):
    # Guardar la estructura de la red y las probabilidades en un archivo JSON
    def convert_to_serializable(obj):
        """Convierte tipos no serializables a tipos compatibles con JSON."""
        if isinstance(obj, (int, float)):
            return obj
        elif hasattr(obj, "tolist"):  # Para ndarrays y otros similares
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)

    data = {
        "edges": list(red.edges()),
        "cpds": []
    }
    for cpd in red.get_cpds():
        cpd_data = {
            "variable": cpd.variable,
            "variable_card": int(cpd.variable_card),  # Convertir a int
            "values": convert_to_serializable(cpd.get_values()),  # Convertir valores a serializable
            "evidence": list(cpd.variables[1:]),  # Excluye la variable en sí misma
            "evidence_card": [int(card) for card in cpd.cardinality[1:]]  # Convertir cardinalidades a int
        }
        data["cpds"].append(cpd_data)

    with open(filename, "w") as file:
        json.dump(data, file)
    print(f"Red bayesiana guardada en {filename}")

def cargar_red_bayesiana(filename):
    # Cargar la estructura de la red y las probabilidades desde un archivo JSON
    with open(filename, "r") as file:
        data = json.load(file)

    red = BayesianNetwork()
    red.add_edges_from(data["edges"])

    cpds = []
    for cpd_data in data["cpds"]:
        cpd = TabularCPD(
            variable=cpd_data["variable"],
            variable_card=cpd_data["variable_card"],
            values=cpd_data["values"],
            evidence=cpd_data["evidence"],
            evidence_card=cpd_data["evidence_card"]
        )
        cpds.append(cpd)

    red.add_cpds(*cpds)
    if red.check_model():
        print("Red bayesiana cargada exitosamente desde el archivo.")
        return red
    else:
        print("La red cargada no es válida.")
        return None

def visualizar_red(red):
    # Visualización de la estructura de la red
    plt.figure(figsize=(10, 6))
    grafo = nx.DiGraph()
    grafo.add_edges_from(red.edges())
    nx.draw_networkx(grafo, with_labels=True, node_size=2000, font_size=10)
    plt.title("Estructura de la Red Bayesiana")
    plt.show()

def inferir_probabilidades(red):
    # Crear el motor de inferencia
    infer = VariableElimination(red)

    # Realizar consultas de probabilidad
    while True:
        consulta = input("\nIngrese el nodo que desea consultar (o 'salir' para finalizar): ").strip()
        if consulta.lower() == 'salir':
            break

        # Consultar probabilidades con o sin evidencia
        evidencia = {}
        agregar_evidencia = input("¿Desea agregar evidencia? (sí/no): ").strip().lower()
        if agregar_evidencia == 'sí':
            while True:
                nodo_ev = input("Ingrese el nodo de la evidencia (o 'salir' para terminar): ").strip()
                if nodo_ev.lower() == 'salir':
                    break
                valor_ev = int(input(f"Ingrese el valor de la evidencia para {nodo_ev}: ").strip())
                evidencia[nodo_ev] = valor_ev

        # Realizar la inferencia
        resultado = infer.query(variables=[consulta], evidence=evidencia)
        print(resultado)

# Programa principal
if __name__ == "__main__":
    print("Bienvenido al programa de creación de redes bayesianas.")
    red = crear_red_bayesiana()
    
    if red:
        # Visualización de la red
        visualizar_red(red)
        
        # Guardar red bayesiana
        guardar = input("¿Desea guardar la red en un archivo? (sí/no): ").strip().lower()
        if guardar == 'sí':
            filename = input("Ingrese el nombre del archivo (con .json): ").strip()
            guardar_red_bayesiana(red, filename)
        
        # Inferencia de probabilidades
        inferir_probabilidades(red)
        
        # Cargar una red desde archivo
        cargar = input("¿Desea cargar una red desde un archivo? (sí/no): ").strip().lower()
        if cargar == 'sí':
            filename = input("Ingrese el nombre del archivo (con .json): ").strip()
            red_cargada = cargar_red_bayesiana(filename)
            if red_cargada:
                inferir_probabilidades(red_cargada)
