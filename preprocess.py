import csv
with open('hate_data_utf.csv', 'r', encoding='utf-8') as input_file,  open('hate_clean.csv', 'w', encoding='utf-8', newline='') as output_file:
    writer=csv.writer(output_file)
    CSV = csv.reader(input_file, delimiter=",")
    for row in CSV:
        if row[5]!="The tweet uses offensive language but not hate speech":
            writer.writerow(row)
