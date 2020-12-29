import os
import random
import torch
import math
import sys
from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

import config
from wrap import wrap_print
from colors import *


def temperature_curve_sample(min_temperature, max_temperature, exponent):
    return (max_temperature - min_temperature) * random.random() ** exponent + min_temperature


def main(args):
    gpt2_config, model_file = config.config[args.model]

    if not os.path.exists(model_file):
        print("Model not found, please download {}".format(args.model))
        print("https://s3.amazonaws.com/models.huggingface.co/bert/{}-pytorch_model.bin".format(args.model))
        raise SystemExit

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    state_dict = torch.load(model_file, map_location=device)
    if args.verbose:
        print(args)

    seed = random.randint(0, 2147483647)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load Model

    enc = get_encoder()
    gpt2_config = GPT2Config(**gpt2_config)
    model = GPT2LMHeadModel(gpt2_config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    torch.cuda.set_device(args.gpu)

    if args.length == -1:
        args.length = gpt2_config.n_ctx // 2
    elif args.length > gpt2_config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % gpt2_config.n_ctx)

    if args.seed:
        if args.rp:
            seed_text = open(args.seed).read().strip()
            gpt2_text = seed_text + "\n\n"
            print(C_USER + seed_text + "\n" + C_ENDC)
            args.seed = None
        else:
            seed_text = open(args.seed).read().strip().replace(args.name + ":", args.gpt2_name + ":").replace(
                args.ai + ":", args.gpt2_ai + ":")

            gpt2_text = seed_text + "\n"
            print(seed_text.replace(args.gpt2_name, C_USER + args.name + C_ENDC).replace(args.gpt2_ai,
                                                                                         C_AI + args.ai + C_ENDC))
            args.seed = None
    else:
        gpt2_text = ""
    if args.log: chatlog = open("chat.log", "a")

    prompt = args.prompt
    init = args.init

    if args.gpt2_ai:
        gpt2_ai = args.gpt2_ai
    else:
        gpt2_ai = args.ai

    # for preventing truncated chat history messing up GPT-2
    if args.rp:
        valid_newlines = ['']
    else:
        valid_newlines = [
            args.gpt2_ai + ": ",
            args.gpt2_name + ": "
        ]


    if args.no_wrap:
        wrap_width = 0
    else:
        wrap_width = args.wrap_width

    if args.rp:
        end_tokens = [enc.encode("\n") + enc.encode("\n"), enc.encode("\n\n"), [enc.encoder['<|endoftext|>']]]
    else:
        end_tokens = [enc.encode("\n"), [enc.encoder['<|endoftext|>']]]

    outputs = {}
    while True:
        finish_sentence = False
        adventure = args.adventure
        if prompt:
            if args.rp:
                wrap_print(prompt, wrap_width)
            else:
                wrap_print(C_USER + args.name + C_ENDC + ": " + prompt, wrap_width)
            user_input = prompt
            prompt = None
        else:
            if args.rp:
                user_input = input(C_USER)
                sys.stdout.write(C_ENDC)
            else:
                user_input = input(C_USER + args.name + C_ENDC + ": ")

            if len(user_input) > 0:
                if user_input.find("|") > 0:  # /puppet forget what this does
                    ind = user_input.find("|")
                    puppet_text = user_input[ind + 1:]
                    outputs[puppet_text.lstrip()] = 1
                    user_input = user_input[:ind]
                    outputs[user_input.lstrip()] = 1
                    if args.rp:
                        gpt2_text += user_input + "\n"
                        gpt2_text += puppet_text + "\n"
                    else:
                        gpt2_text += args.gpt2_name + ": " + user_input + "\n"
                        gpt2_text += args.gpt2_ai + ": " + puppet_text + "\n"
                    continue
                elif user_input.find("~") > 0:  # /puppet forget what this does
                    ind = user_input.find("~")
                    init = user_input[ind + 1:]
                    user_input = user_input[:ind]
                else:
                    # TODO this needs to be replaced with commands
                    if user_input[-1] == "+":  # /question
                        init = random.choice(["How", "What", "How"])
                        user_input = user_input[:-1]
                    elif user_input[-1] == "=":  # /summarize
                        init = random.choice(["So it seems", "Then does that mean", "I think"])
                        user_input = user_input[:-1]
                    elif user_input[-1] == "^":  # /bigthink
                        init = random.choice(
                            ["What if", "I wonder if", "I wonder why", "I wonder whether", "Perhaps we could",
                             "That would work if", "That would work when", "In what ways can we", "In what ways",
                             "If it were possible", "If you could", "How would it be different if"])
                        user_input = user_input[:-1]
                    elif user_input[-1] == "`":  # /adventure
                        adventure = adventure ^ True
                        user_input = user_input[:-1]
                    elif user_input[-1] == ">":  # /autocomplete
                        user_input = user_input[:-1]
                        finish_sentence = True
                        init = user_input
                    elif user_input[-1] == "<":  # /autocomplete adventure mode?
                        user_input = user_input[:-1]
                        finish_sentence = True
                        adventure = True
                    if len(user_input) > 0 and (user_input[0] == "*" or user_input[-1] == "*"):  # /me copycat mode
                        if init == "" and random.random() < 0.8:
                            init = "*"
        if user_input == "" and init == "": break
        if user_input != "":
            if args.rp:
                if finish_sentence:
                    gpt2_msg = user_input
                else:
                    gpt2_msg = user_input + "\n"
            else:
                if finish_sentence:
                    gpt2_msg = args.gpt2_name + ": " + user_input
                else:
                    gpt2_msg = args.gpt2_name + ": " + user_input + "\n"
        else:
            gpt2_msg = ""
        if args.log: chatlog.write(gpt2_msg)
        gpt2_text += gpt2_msg
        original_gpt2_text = gpt2_text
        original_init = init
        outputs[user_input] = 1
        for r in range(1):

            gpt2_text = original_gpt2_text
            init = original_init
            for line in range(random.randint(1, args.chattiness)):
                if not finish_sentence:
                    if not args.rp:
                        gpt2_text += gpt2_ai + ":"
                        if init != "":
                            gpt2_text += " " + init
                    else:
                        if init != "":
                            gpt2_text += init

                context_tokens = enc.encode(gpt2_text)[-args.history_length:]
                gpt2_text = enc.decode(context_tokens)

                valid_context = False
                while not valid_context:
                    for valid_newline in valid_newlines:
                        if gpt2_text[:len(valid_newline)] == valid_newline:
                            valid_context = True
                            break
                    if not valid_context:
                        try:
                            newline_ind = gpt2_text.find("\n")
                            if newline_ind >= 0:
                                gpt2_text = gpt2_text[newline_ind + 1:]
                        except ValueError:
                            print(C_RED + gpt2_text + C_ENDC + "[END]")
                            print(C_YELLOW + "Exception: Invalid chat history")
                            valid_context = True

                            # re-encode fixed chat history
                context_tokens = enc.encode(gpt2_text)[-args.history_length:]
                if args.debug:
                    print(context_tokens)
                    print(C_BLUE + enc.decode(context_tokens) + C_ENDC + "[END]")

                if adventure:
                    choice = 0
                    while choice == 0:
                        potential_outputs = {}
                        for k in range(args.choices):
                            for i in range(args.choices):  # prevent thought loops
                                temperature = temperature_curve_sample(args.min_temperature, args.max_temperature,
                                                                       args.temperature_curve)
                                out = sample_sequence(
                                    model=model, length=args.length,
                                    context=context_tokens if not args.unconditional else None,
                                    start_token=enc.encoder["<|endoftext|>"] if args.unconditional else None,
                                    end_tokens=end_tokens,
                                    batch_size=1,
                                    temperature=temperature,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    device=device,
                                    verbose=args.verbose
                                )
                                out = out[:, len(context_tokens):].tolist()
                                output = enc.decode(out[0]).replace("\r", "").rstrip()
                                if output.lstrip() not in outputs and output.lstrip() not in potential_outputs:
                                    potential_outputs[output.lstrip()] = 1
                                    break
                                if args.debug: print(C_YELLOW + 'Looping...' + C_ENDC)
                        pi = 1
                        potentials = list(potential_outputs.keys())
                        for p in potentials:
                            if finish_sentence:
                                p = " " + p
                            print("{}: {}".format(pi, init + p))
                            pi += 1
                        user_choice = input('Which one? ')
                        if user_choice == '':
                            choice = 0
                        else:
                            choice = min(5, max(0, int(user_choice)))
                    if init == "":
                        output = " " + potentials[choice - 1]
                    else:
                        output = potentials[choice - 1]
                    outputs[output.lstrip()] = 1
                    adventure = False
                else:
                    for i in range(args.choices):  # this should be a function :^)
                        temperature = temperature_curve_sample(args.min_temperature, args.max_temperature,
                                                               args.temperature_curve)
                        out = sample_sequence(
                            model=model, length=args.length,
                            context=context_tokens if not args.unconditional else None,
                            start_token=enc.encoder["<|endoftext|>"] if args.unconditional else None,
                            end_tokens=end_tokens,
                            batch_size=1,
                            temperature=temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            device=device,
                            verbose=args.verbose
                        )
                        out = out[:, len(context_tokens):].tolist()
                        output = enc.decode(out[0]).replace("\r", "").rstrip()

                        if output.lstrip() not in outputs:
                            outputs[output.lstrip()] = 1
                            break
                        if args.debug:
                            print(C_YELLOW + "Looping..." + C_ENDC)
                last_output = output
                eot = output.rfind("<|endoftext|>")
                if eot >= 0:
                    output = output[:eot]
                gpt2_text += output + "\n"
                if init != "": init = " " + init

                if not finish_sentence:
                    if args.rp:
                        wrap_print(init + output, wrap_width)
                    else:
                        wrap_print(C_AI + args.ai + C_ENDC + ":" + init + output, wrap_width)

                    chatlog.write(args.ai + ":" + init + output + "\n")
                else:
                    if args.rp:
                        wrap_print("..." + output, wrap_width)
                    else:
                        wrap_print(C_USER + args.name + C_ENDC + ": ..." + output, wrap_width)
                    chatlog.write(output + "\n")
                if init != "": init = ""
    chatlog.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--prompt", help="initial chat message",default="Hi")
    parser.add_argument("--init", help="initial ai chat message", default="Hello")
    parser.add_argument("--name", help="your name", default="You")
    parser.add_argument("--ai", help="ai name", default="AI")
    parser.add_argument("--seed", help="background text to seed GPT-2 generation from")
    parser.add_argument("--chattiness", type=int, help="how many lines of responses to generate", default=1)
    parser.add_argument("--gpt2-name", help="your gpt-2 name", default="You")
    parser.add_argument("--gpt2-ai", help="ai's gpt-2 name", default="AI")
    parser.add_argument("--no-wrap", action="store_true", help="disable text wrapping")
    parser.add_argument("--wrap-width", "-w", type=int, help="wrap text to character width", default=70)
    parser.add_argument("--model", help="gpt2 model", default="gpt2-medium",
                        choices=["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--verbose", "-v", action="store_true", help="print arguments")
    parser.add_argument("--debug", "-d", action="store_true", help="debugging information")
    parser.add_argument('--unconditional', action='store_true', help="if true, unconditional generation")
    parser.add_argument("--history-length", type=int, default=1000,
                        help="maximum chat context history length in gpt-2 tokens")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--min-temperature", "-t", type=float, default=0.6)
    parser.add_argument("--max-temperature", "-mt", type=float, default=1.0)
    parser.add_argument("--temperature-curve", "-tc", type=float, default=math.e,
                        help="temperature exponent factor of the random temperature sampling")
    parser.add_argument("--choices", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--log", type=bool, default=True, help="save chat log")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU device")
    parser.add_argument("--rp", action="store_true", help="Role playing mode, no names")
    parser.add_argument("--adventure", action="store_true", help="Choose your own adventure mode, provides choices")
    parser.add_argument("--quiet", action="store_true", help="suppress warnings")
    args = parser.parse_args()
    main(args)
