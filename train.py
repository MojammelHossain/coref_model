import util
import math
import torch

if __name__ == "__main__":
	if torch.cuda.is_available():
		device = "cuda"
		for i in range(len(torch.cuda.device_count())):
			print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
	else:
		device = "cpu"

	config = util.initialize_from_env()
	dataloader = util.get_train_dataloader(config)
	model = BertForCoref(config, "cuda").to("cuda")
	bert_optimizer = torch.optim.AdamW([param for name, param in model.named_parameters() if 'bert' in name], lr=config['bert_learning_rate'], eps=config['adam_eps'])
	task_optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'bert' not in name], lr=config['task_learning_rate'], eps=config['adam_eps'])
	print("Parameters:")
	print("bert: {}\ntask: {}".format(len(bert_optimizer.param_groups[0]['params']), len(task_optimizer.param_groups[0]['params'])))

	global_steps = 0
	total_loss = 0
	model.train()
	for epoch in range(1):
	  for batch in dataloader:
	    bert_optimizer.zero_grad()
	    task_optimizer.zero_grad()
	    out, loss = model.forward(batch['input_ids'].to("cuda"), batch['input_mask'].to("cuda"), batch['text_len'],
	                              batch['speaker_ids'].to("cuda"), batch['genre'], True, batch['gold_starts'].to("cuda"),
	                              batch['gold_ends'].to("cuda"), batch['cluster_ids'].to("cuda"), batch['sentence_map'].to("cuda"))
	    total_loss += loss.item()
	    loss.backward()
	    bert_optimizer.step()
	    task_optimizer.step()
	    global_steps += 1