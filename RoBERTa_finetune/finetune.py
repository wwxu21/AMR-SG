import argparse
import glob
import logging
import os
import random
import torch
import sys
sys.path.append("./")
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import convert_examples_to_features, processors, convert_examples_to_features_with_amr

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def feature2list(InputFeature):
    feature_list = []
    for feature_x in InputFeature.choices_features:
        input_ids, input_mask, segment_ids, amr_ids, facts_adj = feature_x['input_ids'], feature_x['input_mask'], feature_x[
            'segment_ids'], feature_x['amr_ids'], feature_x['facts_adj']
        feature_list.append((input_ids, input_mask, segment_ids, amr_ids, facts_adj))
    return feature_list


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.save_steps = t_total // args.save_steps
    args.logging_steps = t_total // args.logging_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate,
        },
        {"params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) ], "weight_decay": 0.0,
         "lr": args.learning_rate, },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * t_total, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for i_epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.amr_flag:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": None,
                    "amr_info": (batch[3], batch[4]),
                    "labels": batch[5],
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": None,
                    "labels": batch[3],
                }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if torch.isnan(loss):
                logger.info("  Bad steps" + str(step))
                assert False
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, epoch_num=i_epoch)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results["eval_acc"] > best_dev_acc:
                            best_dev_acc = results["eval_acc"]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args, model, tokenizer, test=True, epoch_num=i_epoch)
                                for key, value in results_test.items():
                                    tb_writer.add_scalar("test_{}".format(key), value, global_step)
                                logger.info(
                                    "test acc: %s, loss: %s, global steps: %s",
                                    str(results_test["eval_acc"]),
                                    str(results_test["eval_loss"]),
                                    str(global_step),
                                )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, prefix="", test=False, epoch_num=None):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, fact_length=args.fact_length, evaluate=not test, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset)
        # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.amr_flag:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": None,
                        "amr_info": (batch[3], batch[4]),
                        "labels": batch[5],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": None,
                        "labels": batch[3],
                    }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)
        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            writer.write("model           =%s\n" % str(prefix))
            writer.write(
                "total batch size=%d\n"
                % (
                        args.per_gpu_train_batch_size
                        * args.gradient_accumulation_steps
                        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
            )
            writer.write("train num epochs=%d\n" % args.num_train_epochs)
            writer.write("fp16            =%s\n" % args.fp16)
            writer.write("max seq length  =%d\n" % args.max_len)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, fact_length=None, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if task == "race":
        processor = processors[task]()
    else:
        processor = processors[task](args.amr_flag)
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name.split("/"))).pop(),
            str(args.max_len),
            str(task),
        ),
    )
    if args.amr_flag:
        cached_features_file = cached_features_file + "_amr"
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        # features2 = torch.load(cached_features_file[:-4])
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))
        if args.amr_flag:
            features = convert_examples_to_features_with_amr(
                examples,
                label_list,
                args.max_len,
                tokenizer,
                fact_length=fact_length,
                sep_token_extra=False,
                pad_on_left=False,
                pad_token_segment_id=0,
            )
        else:
            features = convert_examples_to_features(
                examples,
                label_list,
                args.max_len,
                tokenizer,
                sep_token_extra=False,
                pad_on_left=False,
                pad_token_segment_id=0,
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    if args.amr_flag:
        all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
        all_amr_ids = torch.tensor(select_field(features, "amr_ids"), dtype=torch.long)
        all_adj = torch.tensor(select_field(features, "facts_adj"), dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_amr_ids, all_adj, all_label_ids)
    else:
        all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        default='race',
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--model_name",
        default='roberta-base',
        type=str,
        required=False,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--data_path",
        default='openbook_no_stopwords',
        type=str,
        required=False,
        help="Path to data",
    )
    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="./cache_file/",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )


    parser.add_argument(
        "--max_len",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", default=True, help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", default=True, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--transition_number", default=3, type=int, help="transition_number in the GRN")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--drop", default=0.1, type=float, help="dropout rate for hidden state.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=4.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio.")

    parser.add_argument("--logging_steps", type=int, default=5000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        # default = True,
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", default=True, help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--amr_flag",
        action="store_true",
        help="Whether to use amr data",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if args.amr_flag:
        from transformers_amr import WEIGHTS_NAME, RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer

    else:
        from transformers import WEIGHTS_NAME, RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    if args.task_name == "race":
        processor = processors[args.task_name]()
    else:
        processor = processors[args.task_name](args.amr_flag)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # proxies = {"http": "http://10.10.1.10:3128","https": "https://10.10.1.10:1080",}
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer

    if args.task_name == 'race':
        args.logging_steps = 10
        args.save_steps = 10
        args.data_dir = './Data/RACE'
        args.output_dir = './saved_models/race_large_ascend'
        config = config_class.from_pretrained(
            args.model_name,
            cache_dir="./cache_file/",
            num_labels=num_labels,
            finetuning_task=args.task_name,
            hidden_dropout_prob=args.drop,
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name,
            cache_dir="./cache_file/",
            do_lower_case=args.do_lower_case,
        )
        model = model_class.from_pretrained(
            args.model_name,
            cache_dir="./cache_file/",
            from_tf=bool(".ckpt" in args.model_name),
            config=config,
        )

    elif args.task_name == 'arc_regents':
        args.logging_steps = 20
        args.save_steps = 20
        args.data_dir = './Data/ARC_Regents_with_context'
        args.output_dir = './saved_models/arc_regents_large_seq256'
        config = config_class.from_pretrained(
            './saved_models/race_large', num_labels=num_labels,
            finetuning_task=args.task_name, )
        tokenizer = tokenizer_class.from_pretrained(
            './saved_models/race_large',
            do_lower_case=args.do_lower_case, )
        model = model_class.from_pretrained('./saved_models/race_large',
                                            config=config, )

    elif args.task_name == 'arc_challenge':
        args.fact_length = 20
        args.logging_steps = 20
        args.save_steps = 20
        pretrained = "race_large"
        args.data_dir = './Data/' + args.data_path
        args.output_dir = './saved_models/' + args.data_path
        tokenizer = tokenizer_class.from_pretrained(
            './saved_models/'+ pretrained,
            do_lower_case=args.do_lower_case, )
        config = config_class.from_pretrained(
            './saved_models/'+ pretrained,
            num_labels=num_labels, finetuning_task=args.task_name,
            pad_token_id=tokenizer.pad_token_id, )
        if args.amr_flag:
            model, log_info = model_class.from_pretrained(
                './saved_models/'+ pretrained, config=config,
                transition_number=args.transition_number, output_loading_info=True)
        else:
            model = model_class.from_pretrained(
                './saved_models/'+ pretrained, config=config)
            special_tokens_dict = {'additional_special_tokens': ["[gcls]",
                                                                 "[gclsr]"]}  # "[gcls]", "[gclsr]" represent the global node and the relation for the global node
            tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))

    elif args.task_name == 'openbook':
        args.fact_length = 15
        args.logging_steps = 20
        args.save_steps = 20
        pretrained = "aristo"
        args.data_dir = './Data/' + args.data_path
        args.output_dir = './saved_models/' + args.data_path
        tokenizer = tokenizer_class.from_pretrained(
            './saved_models/'+ pretrained + '_large',
            do_lower_case=args.do_lower_case, )
        config = config_class.from_pretrained(
            './saved_models/'+ pretrained + '_large',
            num_labels=num_labels, finetuning_task=args.task_name,
            pad_token_id=tokenizer.pad_token_id, )
        if args.amr_flag:
            model, log_info = model_class.from_pretrained(
                './saved_models/'+ pretrained + '_large', config=config,
                transition_number=args.transition_number, output_loading_info=True)

        else:
            model = model_class.from_pretrained(
                './saved_models/'+ pretrained + '_large', config=config)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, fact_length=args.fact_length, evaluate=False)
        # logger.info("evaluate before trainings")
        # result = evaluate(args, model, tokenizer, epoch_num=0)#evaluate before training
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        if args.amr_flag:
            model = model_class.from_pretrained(args.output_dir, transition_number=args.transition_number)
        else:
            model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if args.amr_flag:
                model = model_class.from_pretrained(checkpoint, transition_number=args.transition_number)
            else:
                model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if args.amr_flag:
                model = model_class.from_pretrained(checkpoint, transition_number=args.transition_number)
            else:
                model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, test=True)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
