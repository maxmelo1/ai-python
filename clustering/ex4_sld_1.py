import gower
import pandas as pd

X = pd.DataFrame({
    'Fumante': [1, 0, 0 , 1, 1],
    'Escolaridade': ['Fundamental', 'Médio', 'Nenhuma', 'Nenhuma', 'Superior'],
    'Idade': [18, 23, 32, 27, 36]
})
print(X)
print(gower.gower_matrix(X)[1:, 1:])