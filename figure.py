import matplotlib.pyplot as plt
import json

with open('./model/trainer_state.json', 'r') as f:
  log = json.load(f)

history = log['log_history']

rouge1 = []
rouge2 = []
rougeL = []
x = [1,2,3,4,5]

for data in history:
  if 'eval_rouge1' in data:
    rouge1.append(data['eval_rouge1'])
    rouge2.append(data['eval_rouge2'])
    rougeL.append(data['eval_rougeL'])

plt.plot(x,rouge1, color='red', linestyle="-", linewidth="2", marker=".", label="rouge1")
plt.plot(x,rouge2, color='blue', linestyle="-", linewidth="2", marker=".", label="rouge2")
plt.plot(x,rougeL, color='green', linestyle="-", linewidth="2", marker=".", label="rougeL")

plt.xlabel('Epoch', fontsize="10")
plt.ylabel('Score', fontsize="10")

plt.title('rouge1, rouge2, rougeL', fontsize="18")
plt.legend()
plt.savefig('./image/curve.png')