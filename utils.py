import os
import argparse

from configs import TrainConfig
import imageio


def gif_maker(clients, config):
    for client in clients:
        img_root = client.local_dir + client.store_generated_root
        gif_images = []
        for name_list in os.listdir(img_root):
            if name_list == os.listdir(img_root)[0]:
                for _ in range(config.num_of_clients - 1):
                    gif_images.append(imageio.imread(img_root + name_list))
            elif name_list == os.listdir(img_root).pop():
                for _ in range(config.num_of_clients - 1):
                    gif_images.append(imageio.imread(img_root + name_list))
            else:
                gif_images.append(imageio.imread(img_root + name_list))
        imageio.mimsave(client.local_dir + "Client_" + str(client.id) + ".gif", gif_images, fps=3)


def dir_setup(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main_para_echo(config):
    print("-------------------------------")
    print("Number of clients: {}".format(config.num_of_clients))
    print("Train batch size: {}".format(config.batch_size))
    print("Train epochs: {}".format(config.epochs))
    print("If shuffle: {}".format(config.shuffle))
    print("One communication round contain epochs: {}".format(config.com_epochs))
    print("Using dataset: {}".format(config.dataset))
    print("Generating output images in epochs: {}".format(config.sample_rate))
    print("Using device: {}".format(config.device))
    print("The learning rate: {}".format(config.lr))
    print("-------------------------------")


def parse():
    config = TrainConfig()

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=config.epochs, help='Epochs for the training')
    parser.add_argument("--clients", type=int, default=config.num_of_clients, help='Number of clients')
    parser.add_argument("--shuffle", type=int, default=int(config.shuffle), help='If Shuffle (IID)')
    parser.add_argument("--fed_epochs", type=int, default=config.com_epochs,
                        help='How many epochs for communication round')
    parser.add_argument("--dataset", type=str, default=config.dataset, help='Dataset used for the training')
    parser.add_argument("--check_epochs", type=int, default=config.sample_rate,
                        help='Train process visualization sample rate')
    parser.add_argument("--lr", type=float, default=config.lr, help="Adam: learning rate")

    args = parser.parse_args()

    config.epochs = args.epochs
    config.num_of_clients = args.clients
    config.shuffle = bool(args.shuffle)

    if args.fed_epochs == 0:
        config.com_epochs = args.fed_epochs + 1
    else:
        config.com_epochs = args.fed_epochs

    config.dataset = args.dataset
    config.sample_rate = args.check_epochs
    config.lr = args.lr

    # echo, for linux > command to write to the logs, record the command
    print("python train.py --epochs {} --clients {} --shuffle {} --fed_epochs {} --dataset {} --check_epochs {} --lr {"
          "} > logs.txt".format(args.epochs,
                                args.clients,
                                args.shuffle,
                                args.fed_epochs,
                                args.dataset,
                                args.check_epochs,
                                args.lr))

    # echo
    main_para_echo(config)

    return config


if __name__ == "__main__":
    parse()
    pass
