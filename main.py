from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import yaml

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--method", choices=['DECRAFT(center)', 'DECRAFT(wtp)', 'DECRAFT'], default='DECRAFT(wtp)', type=str)  # specify which method to use
    parser.add_argument("--dataset", choices=['yelp', 'amazon', 'elliptic'], default='amazon', type=str)

    method = vars(parser.parse_args())['method']  # dict
    data_name = vars(parser.parse_args())['dataset']  # dict

    if data_name in ['yelp']:
        yaml_file = "config/yelp_cfg.yaml"
    elif data_name in ['amazon']:
        yaml_file = "config/amazon_cfg.yaml"
    elif data_name in ['elliptic']:
        yaml_file = "config/elliptic_cfg.yaml"
    else:
        raise NotImplementedError("Unsupported method.")

    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args['method'] = method
    args['dataset'] = data_name
    return args


def main(args):
    print(args['method'])
    if args['method'] == ('DECRAFT(center)'):
        from methods.decraft_center.decraft_main import decraft_main, load_decraft_data
        feat_data, labels, train_idx, test_idx, g, cat_features = load_decraft_data(
            args['dataset'], args['test_size'])
        decraft_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features)

    elif args['method'] == ('DECRAFT(wtp)'):
        from methods.decraft_wtp.decraft_main import decraft_main, load_decraft_data
        feat_data, labels, train_idx, test_idx, g, cat_features = load_decraft_data(
            args['dataset'], args['test_size'])
        print('dataset name: {}'.format(args['dataset']))
        decraft_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features)

    else:
        raise NotImplementedError("Unsupported method. ")


if __name__ == "__main__":

    main(parse_args())

