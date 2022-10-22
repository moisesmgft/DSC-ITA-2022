import numpy as np
import csv


dic = {'acao1':np.array([5.0,3121]),'acao2':np.array([231,123])}

def generate_csv(dic):
    append_matrix = []

    for key, value in sorted(dic.items()):
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

generate_csv(dic)