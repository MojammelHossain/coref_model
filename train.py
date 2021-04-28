import os
import util
import time
import math
import torch
import logging
import evaluate
import coref_model

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if torch.cuda.is_available():
        device = "cuda"
        for i in range(torch.cuda.device_count()):
            print("GPU {}: {}\n".format(i, torch.cuda.get_device_name(i)))
    else:
        device = "cpu"
        print("GPU NOT AVAILABLE!!!!\n")

    config = util.initialize_from_env()
    train_dataloader = util.get_dataloader(config)
    model = coref_model.BertForCoref(config, device).to(device)
    bert_optimizer = torch.optim.AdamW([param for name, param in model.named_parameters() if 'bert' in name], lr=config['bert_learning_rate'], eps=config['adam_eps'])
    task_optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if 'bert' not in name], lr=config['task_learning_rate'], eps=config['adam_eps'])
    print("Parameters:")
    print("bert: {}\ntask: {}".format(len(bert_optimizer.param_groups[0]['params']), len(task_optimizer.param_groups[0]['params'])))

    global_steps = 0
    total_loss = 0.0
    max_f1 = 0.0
    
    model.train()
    initial_time = time.time()

    for epoch in range(config['num_epochs']):
        for batch in train_dataloader:
            bert_optimizer.zero_grad()
            task_optimizer.zero_grad()
            out, loss = model.forward(batch['input_ids'].to(device), batch['input_mask'].to(device), batch['text_len'],
                                      batch['speaker_ids'].to(device), batch['genre'], True, batch['gold_starts'].to(device),
                                      batch['gold_ends'].to(device), batch['cluster_ids'].to(device), batch['sentence_map'].to(device))
        total_loss += loss.item()
        loss.backward()
        bert_optimizer.step()
        task_optimizer.step()
        global_steps += 1

        if global_steps % config['report_frequency'] == 0:
            total_time = time.time() - initial_time
            steps_per_second = global_steps/total_time

            avg_loss = total_loss/config['report_frequency']
            logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(global_steps, avg_loss, steps_per_second))
            total_loss = 0.0

        if global_steps > 0 and global_steps % config['eval_frequency'] == 0:
            eval_f1 = evaluate.evaluate(model, config, device)

            path = config["log_dir"]+"/model.{}.pt".format(global_steps)
            torch.save({
                'eval_f1': eval_f1,
                'max_f1' : max_f1,
                'global_steps': global_steps,
                'model': model.state_dict(),
                'bert_optimizer': bert_optimizer.state_dict(),
                'task_optimizer': task_optimizer.state_dict()
                }, path)

            if eval_f1 > max_f1:
                max_f1 = eval_f1
                torch.save({
                    'eval_f1': eval_f1,
                    'max_f1' : max_f1,
                    'global_steps': global_steps,
                    'model': model.state_dict(),
                    'bert_optimizer': bert_optimizer.state_dict(),
                    'task_optimizer': task_optimizer.state_dict()
                    }, (config["log_dir"]+"/model.max.pt"))
