import numpy as np
import csv

append_matrix = []
result = {'acao1':[213,3121],'acao2':[231,123]}
for key, value in sorted(result.items()):
    append_matrix.append([key]+value)

append_matrix = np.array(append_matrix).T

with open("predicao.csv", 'w') as file:
    header = ['Dia']
    for nome in append_matrix[0,:]:
        header.append(nome)
    
    data = []

    i = 1
    for row in append_matrix[1:,:]:
        nrow = [i]
        for share in row:
            nrow.append(share)
        data.append(nrow)
        i += 1
    
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    csvwriter.writerows(data) 