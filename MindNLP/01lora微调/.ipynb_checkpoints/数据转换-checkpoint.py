import csv
import json

csv_file = "merge_shuffle.csv"
json_file = "train.json"
data = []

with open(csv_file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # 读取CSV文件的首行作为header
    for row in csv_reader:
        question = row[0]
        answer = row[1]
        data.append({
            'question': question,
            'answer': answer
        })

with open(json_file, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
