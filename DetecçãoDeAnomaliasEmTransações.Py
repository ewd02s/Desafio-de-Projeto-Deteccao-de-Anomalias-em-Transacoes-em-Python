
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Reprodutibilidade dos resultados
np.random.seed(42)

# 2. Criar transações normais simuladas
quantidade_normal = 500

dados_normais = pd.DataFrame({
    "valor": np.random.normal(
        loc=150,
        scale=50,
        size=quantidade_normal
    ).clip(min=5),

    "horario": np.random.normal(
        loc=14,
        scale=3,
        size=quantidade_normal
    ).clip(min=0, max=23),

    "distancia_km": np.random.exponential(
        scale=5,
        size=quantidade_normal
    )
})

# 3. Criar transações potencialmente anômalas
dados_anomalos = pd.DataFrame({
    "valor": [2500, 3200, 4500, 5000, 7000],
    "horario": [2, 3, 1, 23, 0],
    "distancia_km": [150, 200, 300, 250, 400]
})

# 4. Combinar os dados
dados = pd.concat(
    [dados_normais, dados_anomalos],
    ignore_index=True
)

# 5. Selecionar as características utilizadas pelo modelo
caracteristicas = [
    "valor",
    "horario",
    "distancia_km"
]

X = dados[caracteristicas]

# 6. Criar e treinar o modelo
modelo = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

modelo.fit(X)

# 7. Realizar as previsões
# 1 = normal
# -1 = potencialmente anômala
dados["classificacao"] = modelo.predict(X)

# 8. Calcular o score de anomalia
dados["score_anomalia"] = modelo.decision_function(X)

# 9. Criar uma descrição mais clara
dados["resultado"] = dados["classificacao"].map({
    1: "Normal",
    -1: "Anomalia"
})

# 10. Exibir as transações suspeitas
suspeitas = dados[
    dados["classificacao"] == -1
].sort_values("score_anomalia")

print("===== RESUMO DO PROJETO =====")
print(f"Total de transações: {len(dados)}")
print(f"Transações suspeitas: {len(suspeitas)}")

print("\n===== TRANSAÇÕES SUSPEITAS =====")
print(
    suspeitas[
        [
            "valor",
            "horario",
            "distancia_km",
            "score_anomalia",
            "resultado"
        ]
    ].to_string(index=False)
)

# 11. Visualização dos resultados
plt.figure(figsize=(10, 6))

cores = dados["classificacao"].map({
    1: "blue",
    -1: "red"
})

plt.scatter(
    dados["valor"],
    dados["distancia_km"],
    c=cores,
    alpha=0.6
)

plt.xlabel("Valor da transação (R$)")
plt.ylabel("Distância da transação (km)")
plt.title("Detecção de Anomalias em Transações")
plt.grid(True)

plt.show()