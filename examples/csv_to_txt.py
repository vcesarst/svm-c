# script para transformar csv em txt

import csv
def csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        with open(txt_file, 'w') as txtfile:
            for row in reader:
                txtfile.write(' '.join(row) + '\n')
# Exemplo de uso
csv_file = "iris_scaled.csv" 
txt_file = "iris_scaled.txt"
csv_to_txt(csv_file, txt_file)


